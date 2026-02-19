from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from itertools import zip_longest
from typing import Any, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta, set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, affine_to_spacing, compute_shape_offset, iter_patch, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.networks.utils import meshgrid_ij
from monai.transforms.croppad.array import CenterSpatialCrop, ResizeWithPadOrCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.functional import (
    affine_func,
    flip,
    orientation,
    resize,
    rotate,
    rotate90,
    spatial_resample,
    zoom,
)
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
    resolves_modes,
    scale_affine,
)
from monai.transforms.utils_pytorch_numpy_unification import argsort, argwhere, linalg_inv, moveaxis
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.enums import GridPatchSort, PatchKeys, TraceKeys, TransformBackends
from monai.utils.misc import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, get_equivalent_dtype, get_torch_dtype_from_string

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = [
    "SpatialResample",
    "ResampleToMatch",
    "Spacing",
    "Orientation",
    "Flip",
    "GridDistortion",
    "GridSplit",
    "GridPatch",
    "RandGridPatch",
    "Resize",
    "Rotate",
    "Zoom",
    "Rotate90",
    "RandRotate90",
    "RandRotate",
    "RandFlip",
    "RandGridDistortion",
    "RandAxisFlip",
    "RandZoom",
    "AffineGrid",
    "RandAffineGrid",
    "RandDeformGrid",
    "Resample",
    "Affine",
    "RandAffine",
    "Rand2DElastic",
    "Rand3DElastic",
    "RandSimulateLowResolution",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]





class Zoom(InvertibleTransform, LazyTransform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Sequence[float] | float,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        keep_size: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.zoom = zoom
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.keep_size = keep_size
        self.kwargs = kwargs


    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"edge"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        _zoom = ensure_tuple_rep(self.zoom, img.ndim - 1)  # match the spatial image dim
        _mode = self.mode if mode is None else mode
        _padding_mode = padding_mode or self.padding_mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        lazy_ = self.lazy if lazy is None else lazy
        return zoom(  # type: ignore
            img,
            _zoom,
            self.keep_size,
            _mode,
            _padding_mode,
            _align_corners,
            _dtype,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )




    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)



    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        if transform[TraceKeys.EXTRA_INFO]["do_padcrop"]:
            orig_size = transform[TraceKeys.ORIG_SIZE]
            pad_or_crop = ResizeWithPadOrCrop(spatial_size=orig_size, mode="edge")
            padcrop_xform = transform[TraceKeys.EXTRA_INFO]["padcrop"]
            padcrop_xform[TraceKeys.EXTRA_INFO]["pad_info"][TraceKeys.ID] = TraceKeys.NONE
            padcrop_xform[TraceKeys.EXTRA_INFO]["crop_info"][TraceKeys.ID] = TraceKeys.NONE
            # this uses inverse because spatial_size // 2 in the forward pass of center crop may cause issues
            data = pad_or_crop.inverse_transform(data, padcrop_xform)  # type: ignore
        # Create inverse transform
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        inverse_transform = Resize(spatial_size=transform[TraceKeys.ORIG_SIZE])
        # Apply inverse
        with inverse_transform.trace_transform(False):
            out = inverse_transform(
                data, mode=mode, align_corners=None if align_corners == TraceKeys.NONE else align_corners, dtype=dtype
            )
        return out
        

class RandZoomInverseZoom(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        keep_size: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise ValueError(
                f"min_zoom and max_zoom must have same length, got {len(self.min_zoom)} and {len(self.max_zoom)}."
            )
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.keep_size = keep_size
        self.kwargs = kwargs

        self._zoom: Sequence[float] = [1.0]


    def randomize(self, img: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 1)
        elif len(self._zoom) == 2 and img.ndim > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 2) + ensure_tuple(self._zoom[-1])




    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``,
                ``"area"``}, the interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        # match the spatial image dim
        if randomize:
            self.randomize(img=img)

        lazy_ = self.lazy if lazy is None else lazy
        if not self._do_transform:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        else:
            xform = Zoom(
                self._zoom,
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype,
                lazy=lazy_,
                **self.kwargs,
            )
            # print(self._zoom)
            if len(self._zoom)==3:
                a = (1/self._zoom[0])
                b = (1/self._zoom[1])
                c = (1/self._zoom[2])
            zoom_factor = (a,b,c)
            # print(zoom_factor)
            xformInverse = Zoom(
                zoom_factor,
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype,
                lazy=lazy_,
                **self.kwargs,
            )
            out = xform(img)
            out = xformInverse(out)
        self.push_transform(out, replace=True, lazy=lazy_)
        return out  # type: ignore




    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Zoom(self._zoom).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])





# from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    Affine,
    Flip,
    GridDistortion,
    GridPatch,
    GridSplit,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandAxisFlip,
    RandGridDistortion,
    RandGridPatch,
    RandRotate,
    RandSimulateLowResolution,
    RandZoom,
    ResampleToMatch,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    SpatialResample,
    Zoom,
)
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import





class RandZoomInverseZoomd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dict-based version :py:class:`monai.transforms.RandZoom`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_zoom = RandZoomInverseZoom(
            prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, lazy=lazy, **kwargs
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_zoom.lazy = val
        self._lazy = val


    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandZoomd:
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self




    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_zoom(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d




    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_zoom.inverse(d[key])
        return d





class Rotate(InvertibleTransform, LazyTransform):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = torch.float32,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.angle = angle
        self.keep_size = keep_size
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype


    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        _mode = mode or self.mode
        _padding_mode = padding_mode or self.padding_mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        im_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        output_shape = im_shape if self.keep_size else None
        lazy_ = self.lazy if lazy is None else lazy
        return rotate(  # type: ignore
            img,
            self.angle,
            output_shape,
            _mode,
            _padding_mode,
            _align_corners,
            _dtype,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )




    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)



    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        fwd_rot_mat = transform[TraceKeys.EXTRA_INFO]["rot_mat"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        inv_rot_mat = linalg_inv(convert_to_numpy(fwd_rot_mat))

        _, _m, _p, _ = resolves_modes(mode, padding_mode)
        xform = AffineTransform(
            normalized=False,
            mode=_m,
            padding_mode=_p,
            align_corners=False if align_corners == TraceKeys.NONE else align_corners,
            reverse_indexing=True,
        )
        img_t: torch.Tensor = convert_data_type(data, MetaTensor, dtype=dtype)[0]
        transform_t, *_ = convert_to_dst_type(inv_rot_mat, img_t)
        sp_size = transform[TraceKeys.ORIG_SIZE]
        out: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=sp_size).float().squeeze(0)
        out = convert_to_dst_type(out, dst=data, dtype=out.dtype)[0]
        if isinstance(out, MetaTensor):
            affine = convert_to_tensor(out.peek_pending_affine(), track_meta=False)
            mat = to_affine_nd(len(affine) - 1, transform_t)
            out.affine @= convert_to_dst_type(mat, affine)[0]
        return out


class RandRotateInverseRotate(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly rotate the input arrays.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y). only work for 3D data.
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z). only work for 3D data.
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Rotate.backend

    def __init__(
        self,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = np.float32,
        lazy: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])



    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ):
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize()

        lazy_ = self.lazy if lazy is None else lazy
        if self._do_transform:
            ndim = len(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:])
            rotator = Rotate(
                angle=self.x if ndim == 2 else (self.x, self.y, self.z),
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype or img.dtype,
                lazy=lazy_,
            )
            inverse_rotator = Rotate(
                angle= -1*self.x if ndim == 2 else (-1*self.x, -1*self.y, -1*self.z),
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype or img.dtype,
                lazy=lazy_,
            )
            out = rotator(img)
            out = inverse_rotator(out)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        self.push_transform(out, replace=True, lazy=lazy_)
        return out




    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Rotate(0).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])





# from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    Affine,
    Flip,
    GridDistortion,
    GridPatch,
    GridSplit,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandAxisFlip,
    RandGridDistortion,
    RandGridPatch,
    RandRotate,
    RandSimulateLowResolution,
    RandZoom,
    ResampleToMatch,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    SpatialResample,
    Zoom,
)
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import

class RandRotateInverseRotated(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate`
    Randomly rotates the input arrays.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y). only work for 3D data.
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z). only work for 3D data.
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = RandRotate.backend

    def __init__(
        self,
        keys: KeysCollection,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_rotate = RandRotateInverseRotate(
            range_x=range_x, range_y=range_y, range_z=range_z, prob=1.0, keep_size=keep_size, lazy=lazy
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_rotate.lazy = val
        self._lazy = val


    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandRotated:
        super().set_random_state(seed, state)
        self.rand_rotate.set_random_state(seed, state)
        return self




    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)

        # all the keys share the same random rotate angle
        self.rand_rotate.randomize()
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_rotate(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d




    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_rotate.inverse(d[key])
        return d
