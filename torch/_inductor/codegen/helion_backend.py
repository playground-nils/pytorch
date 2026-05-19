from __future__ import annotations

import contextlib
import hashlib
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..utils import get_fused_kernel_name, get_kernel_metadata, IndentedBuffer
from ..virtualized import V
from .common import BackendFeature, CSEVariable, DeferredLine, OpOverrides, RemovedArg
from .simd import SIMDKernel, SIMDScheduling

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import sympy

    from ..ir import IRNode
    from ..scheduler import BaseSchedulerNode
    from .simd_kernel_features import SIMDKernelFeatures

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class HelionKernelWrapper:
    def __init__(self, kernel_fn: Callable[..., Any], kernel_source: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_source = kernel_source

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.kernel_fn(*args)


class HelionKernelOverrides(OpOverrides):
    @staticmethod
    def sin(x):
        return f"torch.sin({x})"

    @staticmethod
    def cos(x):
        return f"torch.cos({x})"

    @staticmethod
    def tan(x):
        return f"torch.tan({x})"

    @staticmethod
    def sinh(x):
        return f"torch.sinh({x})"

    @staticmethod
    def cosh(x):
        return f"torch.cosh({x})"

    @staticmethod
    def tanh(x):
        return f"torch.tanh({x})"

    @staticmethod
    def asin(x):
        return f"torch.asin({x})"

    @staticmethod
    def acos(x):
        return f"torch.acos({x})"

    @staticmethod
    def atan(x):
        return f"torch.atan({x})"

    @staticmethod
    def exp(x):
        return f"torch.exp({x})"

    @staticmethod
    def exp2(x):
        return f"torch.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"torch.expm1({x})"

    @staticmethod
    def log(x):
        return f"torch.log({x})"

    @staticmethod
    def log2(x):
        return f"torch.log2({x})"

    @staticmethod
    def log10(x):
        return f"torch.log10({x})"

    @staticmethod
    def log1p(x):
        return f"torch.log1p({x})"

    @staticmethod
    def sqrt(x):
        return f"torch.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"torch.rsqrt({x})"

    @staticmethod
    def abs(x):
        return f"torch.abs({x})"

    @staticmethod
    def sigmoid(x):
        return f"torch.sigmoid({x})"

    @staticmethod
    def relu(x):
        return f"torch.relu({x})"

    @staticmethod
    def floor(x):
        return f"torch.floor({x})"

    @staticmethod
    def ceil(x):
        return f"torch.ceil({x})"

    @staticmethod
    def trunc(x):
        return f"torch.trunc({x})"

    @staticmethod
    def round(x):
        return f"torch.round({x})"

    @staticmethod
    def sign(x):
        return f"torch.sign({x})"

    @staticmethod
    def reciprocal(x):
        return f"torch.reciprocal({x})"

    @staticmethod
    def erf(x):
        return f"torch.erf({x})"

    @staticmethod
    def erfc(x):
        return f"torch.erfc({x})"

    @staticmethod
    def erfinv(x):
        return f"torch.erfinv({x})"

    @staticmethod
    def lgamma(x):
        return f"torch.lgamma({x})"

    @staticmethod
    def isnan(x):
        return f"torch.isnan({x})"

    @staticmethod
    def isinf(x):
        return f"torch.isinf({x})"

    @staticmethod
    def isfinite(x):
        return f"torch.isfinite({x})"

    @staticmethod
    def where(cond, a, b):
        return f"torch.where({cond}, {a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"torch.maximum({a}, {b})"

    @staticmethod
    def minimum(a, b):
        return f"torch.minimum({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"torch.pow({a}, {b})"

    @staticmethod
    def atan2(a, b):
        return f"torch.atan2({a}, {b})"

    @staticmethod
    def fmod(a, b):
        return f"torch.fmod({a}, {b})"

    @staticmethod
    def remainder(a, b):
        return f"torch.remainder({a}, {b})"

    @staticmethod
    def masked(mask, body, other):
        result = body()
        if isinstance(other, float):
            if math.isnan(other):
                other_str = "float('nan')"
            elif math.isinf(other):
                other_str = "float('inf')" if other > 0 else "float('-inf')"
            else:
                other_str = repr(other)
        else:
            other_str = repr(other)
        return f"torch.where({mask}, {result}, {other_str})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        return f"({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        return f"({x}).view({_torch_dtype_str(dtype)})"

    @staticmethod
    def constant(value, dtype):
        if isinstance(value, float):
            if math.isnan(value):
                return "float('nan')"
            if math.isinf(value):
                return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)

    @staticmethod
    def index_expr(expr, dtype):
        if expr.is_number:
            return repr(int(expr))
        # Try to evaluate to a concrete value using known shape bindings
        try:
            val = int(expr)
            return repr(val)
        except (TypeError, ValueError):
            pass
        return V.kernel.helion_index_expr(expr)

    @staticmethod
    def frexp(x):
        return (f"torch.frexp({x}).mantissa", f"torch.frexp({x}).exponent")

    @staticmethod
    def trunc_to_int(x, dtype):
        return f"({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def ceil_to_int(x, dtype):
        return f"torch.ceil({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def floor_to_int(x, dtype):
        return f"torch.floor({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def round_to_int(x, dtype):
        return f"torch.round({x}).to({_torch_dtype_str(dtype)})"

    # Logical operators
    @staticmethod
    def logical_and(a, b):
        return f"torch.logical_and({a}, {b})"

    @staticmethod
    def logical_or(a, b):
        return f"torch.logical_or({a}, {b})"

    @staticmethod
    def logical_not(a):
        return f"torch.logical_not({a})"

    @staticmethod
    def signbit(x):
        return f"torch.signbit({x})"

    @staticmethod
    def truncdiv(a, b):
        return f"torch.div({a}, {b}, rounding_mode='trunc')"

    @staticmethod
    def floordiv(a, b):
        return f"torch.div({a}, {b}, rounding_mode='floor')"

    @staticmethod
    def nextafter(a, b):
        return f"torch.nextafter({a}, {b})"

    @staticmethod
    def fma(a, b, c):
        return f"({a} * {b} + {c})"

    @staticmethod
    def mul_rn(a, b):
        return f"({a} * {b})"

    # torch.special functions
    @staticmethod
    def digamma(x):
        return f"torch.digamma({x})"

    @staticmethod
    def i0(x):
        return f"torch.special.i0({x})"

    @staticmethod
    def i0e(x):
        return f"torch.special.i0e({x})"

    @staticmethod
    def i1(x):
        return f"torch.special.i1({x})"

    @staticmethod
    def i1e(x):
        return f"torch.special.i1e({x})"

    @staticmethod
    def ndtr(x):
        return f"torch.special.ndtr({x})"

    @staticmethod
    def ndtri(x):
        return f"torch.special.ndtri({x})"

    @staticmethod
    def log_ndtr(x):
        return f"torch.special.log_ndtr({x})"

    @staticmethod
    def erfcx(x):
        return f"torch.special.erfcx({x})"

    @staticmethod
    def igamma(a, x):
        return f"torch.special.gammainc({a}, {x})"

    @staticmethod
    def igammac(a, x):
        return f"torch.special.gammaincc({a}, {x})"

    @staticmethod
    def gammainc(a, x):
        return f"torch.special.gammainc({a}, {x})"

    @staticmethod
    def gammaincc(a, x):
        return f"torch.special.gammaincc({a}, {x})"

    @staticmethod
    def polygamma(n, x):
        return f"torch.special.polygamma({n}, {x})"

    @staticmethod
    def zeta(x, q):
        return f"torch.special.zeta({x}, {q})"

    # Bessel functions
    @staticmethod
    def bessel_j0(x):
        return f"torch.special.bessel_j0({x})"

    @staticmethod
    def bessel_j1(x):
        return f"torch.special.bessel_j1({x})"

    @staticmethod
    def bessel_y0(x):
        return f"torch.special.bessel_y0({x})"

    @staticmethod
    def bessel_y1(x):
        return f"torch.special.bessel_y1({x})"

    @staticmethod
    def modified_bessel_i0(x):
        return f"torch.special.modified_bessel_i0({x})"

    @staticmethod
    def modified_bessel_i1(x):
        return f"torch.special.modified_bessel_i1({x})"

    @staticmethod
    def modified_bessel_k0(x):
        return f"torch.special.modified_bessel_k0({x})"

    @staticmethod
    def modified_bessel_k1(x):
        return f"torch.special.modified_bessel_k1({x})"

    @staticmethod
    def scaled_modified_bessel_k0(x):
        return f"torch.special.scaled_modified_bessel_k0({x})"

    @staticmethod
    def scaled_modified_bessel_k1(x):
        return f"torch.special.scaled_modified_bessel_k1({x})"

    @staticmethod
    def spherical_bessel_j0(x):
        return f"torch.special.spherical_bessel_j0({x})"

    @staticmethod
    def airy_ai(x):
        return f"torch.special.airy_ai({x})"

    # Chebyshev polynomials
    @staticmethod
    def chebyshev_polynomial_t(x, n):
        return f"torch.special.chebyshev_polynomial_t({x}, {n})"

    @staticmethod
    def chebyshev_polynomial_u(x, n):
        return f"torch.special.chebyshev_polynomial_u({x}, {n})"

    @staticmethod
    def chebyshev_polynomial_v(x, n):
        return f"torch.special.chebyshev_polynomial_v({x}, {n})"

    @staticmethod
    def chebyshev_polynomial_w(x, n):
        return f"torch.special.chebyshev_polynomial_w({x}, {n})"

    @staticmethod
    def legendre_polynomial_p(x, n):
        return f"torch.special.legendre_polynomial_p({x}, {n})"

    @staticmethod
    def shifted_chebyshev_polynomial_t(x, n):
        return f"torch.special.shifted_chebyshev_polynomial_t({x}, {n})"

    @staticmethod
    def shifted_chebyshev_polynomial_u(x, n):
        return f"torch.special.shifted_chebyshev_polynomial_u({x}, {n})"

    @staticmethod
    def shifted_chebyshev_polynomial_v(x, n):
        return f"torch.special.shifted_chebyshev_polynomial_v({x}, {n})"

    @staticmethod
    def shifted_chebyshev_polynomial_w(x, n):
        return f"torch.special.shifted_chebyshev_polynomial_w({x}, {n})"

    @staticmethod
    def hermite_polynomial_h(x, n):
        return f"torch.special.hermite_polynomial_h({x}, {n})"

    @staticmethod
    def hermite_polynomial_he(x, n):
        return f"torch.special.hermite_polynomial_he({x}, {n})"

    @staticmethod
    def laguerre_polynomial_l(x, n):
        return f"torch.special.laguerre_polynomial_l({x}, {n})"


def _torch_dtype_str(dtype: torch.dtype) -> str:
    s = str(dtype)
    return s if s.startswith("torch.") else f"torch.{s}"


HELION_KERNEL_NAME = "_inductor_helion_kernel"

REDUCTION_TYPE_MAP = {
    "sum": "torch.sum",
    "max": "torch.amax",
    "amax": "torch.amax",
    "min": "torch.amin",
    "amin": "torch.amin",
    "prod": "torch.prod",
    "any": "torch.any",
    "argmax": "torch.argmax",
    "argmin": "torch.argmin",
}

ATOMIC_OP_FOR_REDUCTION = {
    "sum": "hl.atomic_add",
    "amax": "hl.atomic_max",
    "max": "hl.atomic_max",
    "amin": "hl.atomic_min",
    "min": "hl.atomic_min",
}

INIT_FOR_REDUCTION = {
    "sum": ".zero_()",
    "amax": ".fill_(float('-inf'))",
    "max": ".fill_(float('-inf'))",
    "amin": ".fill_(float('inf'))",
    "min": ".fill_(float('inf'))",
}


class HelionKernel(SIMDKernel):
    overrides = HelionKernelOverrides  # type: ignore[assignment]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_lines: list[str] = []
        self._output_ndim: int = 0
        self._reduction_type: str | None = None
        # Maps buffer name to (size, stride, offset) for reinterpret_tensor.
        self._input_reinterprets: dict[str, tuple[tuple[Any, ...], tuple[Any, ...], Any]] = {}
        # Record load/store index expressions for stride analysis.
        self._load_indices: dict[str, Any] = {}
        self._store_index: Any = None

    def _has_reduction(self) -> bool:
        return any(
            tree.is_reduction and tree.numel != 1
            for tree in self.range_trees
        )

    def check_bounds(self, expr, size, lower, upper):
        # Helion handles bounds checking internally via tile masking
        pass

    def helion_index_expr(self, expr) -> str:
        """Convert a sympy index expression to Helion tile-based code.

        Maps iteration range variables (x0, y0, z0, etc.) to their
        corresponding tile variable's .index property which provides the
        actual integer positions within that tile.
        """
        import sympy as sp

        from .simd import prefix_is_reduction

        # Build mapping from sympy symbols to tile variable index expressions
        replacements: dict[sp.Symbol, str] = {}
        for sym, entry in self.range_tree_nodes.items():
            tree = entry.parent
            if prefix_is_reduction(tree.prefix):
                # Reduction variables don't need explicit index expressions
                # in Helion since reductions are handled by torch reduction ops
                continue
            # Map pointwise range tree prefixes to tile variable names
            # Range trees are ordered: z (dim 0), y (dim 1), x (dim 2) for 3D
            # or y (dim 0), x (dim 1) for 2D, or x (dim 0) for 1D
            # We need to figure out which tile_N this prefix corresponds to
            tile_var = self._prefix_to_tile_var(tree.prefix)
            if tile_var is not None:
                replacements[sym] = f"{tile_var}.index"

        if not replacements:
            return str(expr)

        # If expression is a single symbol, return its replacement directly
        if expr in replacements:
            return replacements[expr]

        # For compound expressions, substitute symbols with tile index refs
        # Use sympy substitution to handle the expression correctly
        result = str(expr)
        # Sort by symbol name length (longest first) to avoid partial replacements
        for sym, tile_expr in sorted(
            replacements.items(), key=lambda x: -len(str(x[0]))
        ):
            result = result.replace(str(sym), tile_expr)
        return result

    def _prefix_to_tile_var(self, prefix: str) -> str | None:
        """Map a range tree prefix to its Helion tile variable name."""
        ndim = self._output_ndim
        if ndim <= 1:
            return "tile"

        # For ND kernels, the pointwise range trees are ordered z, y, x
        # (from all_prefixes = ["z", "y", "x", "r0_", "r1_"])
        # and they map to tile_0, tile_1, tile_2, ...
        # Count pointwise trees and their ordering
        pointwise_prefixes = []
        for tree in self.range_trees:
            if not tree.prefix.startswith("r"):
                pointwise_prefixes.append(tree.prefix)

        if prefix in pointwise_prefixes:
            idx = pointwise_prefixes.index(prefix)
            return f"tile_{idx}"
        return None

    def _get_buffer_ndim(self, name: str) -> int:
        """Get the ndim of a buffer by its internal name."""
        buf = V.graph.try_get_buffer(name)
        if buf is not None:
            return len(buf.get_size())
        return self._output_ndim

    def _tile_index(self, name: str, index: sympy.Expr) -> str:
        """Return the appropriate tile indexing string for a buffer access.

        Uses right-aligned broadcasting: a buffer with ndim M uses the last M
        tile variables from the output's N tile variables.
        For reduction buffers, the last dim is accessed via ':' (the reduction
        dim), and the remaining dims use right-aligned tile variables.
        """
        out_ndim = self._output_ndim
        buf_ndim = self._get_buffer_ndim(name)
        has_reduction = self._has_reduction()

        # Scalar buffer (0-dim): no indexing needed
        if buf_ndim == 0:
            return "()"

        # Full reduction to scalar output (out_ndim == 0): the tile loop covers
        # the reduction dimension directly, so just use "tile" for all accesses.
        if out_ndim == 0:
            return "tile"

        # Check if index references any reduction symbols
        uses_reduction = False
        if has_reduction:
            for symbol in index.free_symbols:
                if symbol in self.range_tree_nodes:
                    entry = self.range_tree_nodes[symbol]
                    if entry.parent.prefix.startswith("r"):
                        uses_reduction = True
                        break

        if out_ndim <= 1 and not uses_reduction:
            return "tile"

        if out_ndim <= 1:
            if uses_reduction and self.inside_reduction:
                return "tile, :"
            return "tile"

        # Multi-dim case: right-aligned broadcasting
        if uses_reduction and self.inside_reduction:
            # Buffer has reduction dim as its last dim; tile the pointwise part
            pw_ndim = buf_ndim - 1
            start = max(0, out_ndim - pw_ndim)
            tile_parts = [f"tile_{i}" for i in range(start, out_ndim)]
            tile_parts.append(":")
        else:
            start = max(0, out_ndim - buf_ndim)
            tile_parts = [f"tile_{i}" for i in range(start, out_ndim)]

        if len(tile_parts) == 1:
            return tile_parts[0]
        return ", ".join(tile_parts)

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buf = self.args.input(name)
        idx = self._tile_index(name, index)
        if name in self._load_indices:
            if self._load_indices[name] is not None and self._load_indices[name] != index:
                self._load_indices[name] = None
        else:
            self._load_indices[name] = index
        return self.cse.generate(self.loads, f"{buf}[{idx}]")

    def _compute_input_reinterprets(self) -> None:
        """Detect inputs needing reinterpretation for correct tile indexing.

        When iteration is flat (1D range tree) but the output is multi-dim,
        the kernel tiles over out_ptr0.size() and indexes all buffers with the
        same tile variables. We derive the correct view for each input from
        the relationship between load and store index expressions.
        """
        out_ndim = self._output_ndim
        if out_ndim < 2:
            return
        if self._store_index is None:
            return

        pw_trees = [t for t in self.range_trees if not t.prefix.startswith("r")]
        pw_tree_count = len(pw_trees)

        if pw_tree_count == 1:
            self._compute_reinterprets_flat()
        elif pw_tree_count == out_ndim:
            self._compute_reinterprets_nd(pw_trees)

    def _compute_reinterprets_nd(self, pw_trees: list[Any]) -> None:
        """Handle reinterprets when multiple range trees map to output dims.

        In the multi-tree case, hl.tile(out_ptr.size()) makes tile_i iterate
        over out_ptr.size(i). For in_ptr[tile_0, tile_1], Helion computes the
        flat offset as in_ptr.stride(0)*tile_0 + in_ptr.stride(1)*tile_1.
        We determine the correct input stride by matching the store index
        coefficients to output strides (to find the symbol-to-dim mapping),
        then extracting load index coefficients in output dimension order.
        """
        out_ndim = self._output_ndim
        hint = V.graph.sizevars.optimization_hint

        # Get the output buffer's actual size and stride
        out_size: list[int] | None = None
        out_stride: list[int] | None = None
        for buf_name, val in self.args.output_buffers.items():
            if isinstance(val, RemovedArg):
                continue
            buf = V.graph.try_get_buffer(buf_name)
            if buf is not None:
                out_size = [hint(s, fallback=0) for s in buf.get_size()]
                out_stride = [hint(s, fallback=0) for s in buf.get_stride()]
                break
        if out_size is None:
            for buf_name in self.args.inplace_buffers:
                buf = V.graph.try_get_buffer(buf_name)
                if buf is not None:
                    out_size = [hint(s, fallback=0) for s in buf.get_size()]
                    out_stride = [hint(s, fallback=0) for s in buf.get_stride()]
                    break
        if out_size is None or out_stride is None or len(out_size) != out_ndim:
            return

        # Get the iteration symbol for each tree
        tree_syms = []
        for tree in pw_trees:
            sym = None
            for s, entry in self.range_tree_nodes.items():
                if entry.parent is tree:
                    sym = s
                    break
            if sym is None:
                return
            tree_syms.append(sym)

        # Determine which tree symbol maps to which output dimension.
        # From the store index, find the coefficient of each tree symbol.
        # Match each coefficient to an output stride to get the dim mapping.
        store_coeffs = {}
        for sym in tree_syms:
            coeff = self._store_index.coeff(sym)
            if not coeff.is_number:
                return
            store_coeffs[sym] = int(coeff)

        # Build sym -> output_dim mapping by matching store coefficients
        # to output strides. Each store coeff should equal some out_stride[d].
        sym_to_dim: dict[Any, int] = {}
        used_dims: set[int] = set()
        for sym in tree_syms:
            sc = store_coeffs[sym]
            matched = False
            for d in range(out_ndim):
                if d not in used_dims and out_stride[d] == sc:
                    sym_to_dim[sym] = d
                    used_dims.add(d)
                    matched = True
                    break
            if not matched:
                return
        if len(sym_to_dim) != out_ndim:
            return

        for buf_name in self.args.input_buffers:
            if buf_name not in self._load_indices:
                continue
            load_index = self._load_indices[buf_name]
            if load_index is None:
                continue
            if self._get_buffer_ndim(buf_name) != out_ndim:
                continue

            # Extract load index coefficient for each tree symbol,
            # then place it at the corresponding output dimension.
            in_strides = [0] * out_ndim
            valid = True
            for sym in tree_syms:
                load_coeff = load_index.coeff(sym)
                if not load_coeff.is_number:
                    valid = False
                    break
                in_strides[sym_to_dim[sym]] = int(load_coeff)
            if not valid:
                continue

            in_size = tuple(out_size)
            in_stride = tuple(in_strides)

            # Skip if the buffer already has the output's size. Helion
            # handles stride differences natively via tile indexing.
            buf = V.graph.try_get_buffer(buf_name)
            if buf is not None:
                orig_size = [hint(s, fallback=0) for s in buf.get_size()]
                if list(orig_size) == list(in_size):
                    continue
            else:
                gi = V.graph.graph_inputs.get(buf_name)
                if gi is not None:
                    orig_size = [hint(s, fallback=0) for s in gi.get_size()]
                    if list(orig_size) == list(in_size):
                        continue

            # Compute offset
            load_offset = load_index
            for sym in tree_syms:
                load_offset = load_offset.subs(sym, 0)
            offset_val = int(load_offset) if load_offset.is_number else 0

            self._input_reinterprets[buf_name] = (in_size, in_stride, offset_val)

    def _compute_reinterprets_flat(self) -> None:
        """Handle reinterprets when iteration is flattened to a single range tree."""
        out_ndim = self._output_ndim
        hint = V.graph.sizevars.optimization_hint

        # Find the output buffer's size and stride
        out_size: list[int] | None = None
        out_stride: list[int] | None = None
        for buf_name, val in self.args.output_buffers.items():
            if isinstance(val, RemovedArg):
                continue
            buf = V.graph.try_get_buffer(buf_name)
            if buf is not None:
                out_size = [hint(s, fallback=0) for s in buf.get_size()]
                out_stride = [hint(s, fallback=0) for s in buf.get_stride()]
                break
        if out_size is None:
            for buf_name in self.args.inplace_buffers:
                buf = V.graph.try_get_buffer(buf_name)
                if buf is not None:
                    out_size = [hint(s, fallback=0) for s in buf.get_size()]
                    out_stride = [hint(s, fallback=0) for s in buf.get_stride()]
                    break
        if out_size is None or out_stride is None or len(out_size) != out_ndim:
            return

        # Collect all pointwise iteration symbols from the store index,
        # ordered by their coefficient (largest first = outermost dimension).
        store_iter_syms: list[Any] = []
        for s in self._store_index.free_symbols:
            if s in self.range_tree_nodes:
                entry = self.range_tree_nodes[s]
                if not entry.parent.prefix.startswith("r"):
                    coeff = self._store_index.coeff(s)
                    if coeff.is_number and int(coeff) != 0:
                        store_iter_syms.append((s, int(coeff)))
        # Sort by coefficient descending (outermost dim has largest stride)
        store_iter_syms.sort(key=lambda x: -x[1])

        if len(store_iter_syms) == 0:
            return

        store_sym_set = {s for s, _ in store_iter_syms}
        # Use multi-symbol path when #symbols matches output ndim
        use_per_dim = len(store_iter_syms) == out_ndim

        for buf_name in self.args.input_buffers:
            if buf_name not in self._load_indices:
                continue
            load_index = self._load_indices[buf_name]
            if load_index is None:
                continue
            if self._get_buffer_ndim(buf_name) != out_ndim:
                continue

            # Collect iteration symbols used in the load index
            load_iter_syms = set()
            for s in load_index.free_symbols:
                if s in self.range_tree_nodes:
                    entry = self.range_tree_nodes[s]
                    if not entry.parent.prefix.startswith("r"):
                        load_iter_syms.add(s)

            # Skip if load uses different symbols than the store
            if load_iter_syms != store_sym_set:
                continue

            if use_per_dim:
                # Multi-symbol: extract per-dimension stride directly
                in_strides = []
                valid = True
                for sym, _store_coeff in store_iter_syms:
                    load_coeff = load_index.coeff(sym)
                    if not load_coeff.is_number:
                        valid = False
                        break
                    in_strides.append(int(load_coeff))
                if not valid:
                    continue
                in_stride = tuple(in_strides)
            else:
                # Single-symbol: derive via scale factor on output strides
                sym, store_coeff_val = store_iter_syms[0]
                load_coeff = load_index.coeff(sym)
                if not load_coeff.is_number:
                    continue
                load_coeff_val = int(load_coeff)
                if store_coeff_val == 0:
                    continue
                if load_coeff_val % store_coeff_val != 0:
                    continue
                scale = load_coeff_val // store_coeff_val
                in_stride = tuple(scale * st for st in out_stride)

            in_size = tuple(out_size)
            in_stride = tuple(in_stride) if not isinstance(in_stride, tuple) else in_stride

            # Skip if the buffer already has the output's size. Helion
            # handles stride differences natively (tile indexing respects
            # the tensor's own strides), so only a size mismatch requires
            # reinterpret_tensor.
            buf = V.graph.try_get_buffer(buf_name)
            if buf is not None:
                orig_size = [hint(s, fallback=0) for s in buf.get_size()]
                if list(orig_size) == list(in_size):
                    continue
            else:
                gi = V.graph.graph_inputs.get(buf_name)
                if gi is not None:
                    orig_size = [hint(s, fallback=0) for s in gi.get_size()]
                    if list(orig_size) == list(in_size):
                        continue

            # Compute offset
            load_offset = load_index
            store_offset = self._store_index
            for sym, _ in store_iter_syms:
                load_offset = load_offset.subs(sym, 0)
                store_offset = store_offset.subs(sym, 0)
            load_offset_val = int(load_offset) if load_offset.is_number else 0
            store_offset_val = int(store_offset) if store_offset.is_number else 0
            offset_val = load_offset_val - store_offset_val

            self._input_reinterprets[buf_name] = (in_size, in_stride, offset_val)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:
        out = self.args.output(name)
        self.store_buffer_names.add(name)
        if self._store_index is None:
            self._store_index = index
        idx = self._tile_index(name, index)
        # If this is a reduction kernel and the store doesn't include the
        # reduction dim (no ':'), squeeze the value to remove keepdim.
        # Skip for scalar outputs (ndim==0) which already reduce without keepdim.
        if self._has_reduction() and ":" not in idx and self._output_ndim > 0:
            value = self.cse.generate(
                self.compute, f"({value}).squeeze(-1)"
            )
        if mode == "atomic_add":
            line = f"hl.atomic_add({out}, ({idx},), {value})"
        elif (
            self._output_ndim == 0
            and self._has_reduction()
            and self._reduction_type in ATOMIC_OP_FOR_REDUCTION
        ):
            atomic_fn = ATOMIC_OP_FOR_REDUCTION[self._reduction_type]
            line = f"{atomic_fn}({out}, {idx}, {value})"
        else:
            line = f"{out}[{idx}] = {value}"
        self.store_lines.append(DeferredLine(name, line))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """Handle reduction ops by emitting torch reduction calls."""
        if reduction_type == "welford_reduce":
            return self.welford_reduce_fallback(dtype, value)
        self._reduction_type = reduction_type
        reduction_fn = REDUCTION_TYPE_MAP[reduction_type]
        if self._output_ndim == 0:
            # Full reduction to scalar: reduce all elements, no keepdim
            expr = f"{reduction_fn}({value})"
        else:
            expr = f"{reduction_fn}({value}, dim=-1, keepdim=True)"
        return self.cse.generate(self.compute, expr)

    def store_reduction(self, name, index, value):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx():
            if not self._has_reduction():
                yield
                return
            prior = self.inside_reduction
            self.inside_reduction = False
            try:
                yield
            finally:
                self.inside_reduction = prior

        return ctx()

    def codegen_kernel(self) -> str:
        buf = IndentedBuffer()
        buf.writeline("import torch")
        buf.writeline("import helion")
        buf.writeline("import helion.language as hl")
        buf.writeline("from helion.runtime.settings import Settings")
        buf.writeline("")
        buf.writeline("")

        arg_defs, _, _, _ = self.args.python_argdefs()
        param_names = [a.name for a in arg_defs]
        param_strs = [f"{name}: torch.Tensor" for name in param_names]

        backend = _get_helion_backend()
        settings_parts = [f"backend='{backend}'"]
        if backend == "pallas":
            device = V.graph.get_current_device_or_throw()
            if device.type == "cpu":
                settings_parts.append("pallas_interpret=True")
        settings_parts.append(f"autotune_effort='{config.helion_autotune_effort}'")
        if config.helion_print_output_code:
            settings_parts.append("print_output_code=True")
        settings_str = ", ".join(settings_parts)
        buf.writeline(f"@helion.kernel(settings=Settings({settings_str}))")
        buf.writeline(
            f"def {HELION_KERNEL_NAME}({', '.join(param_strs)}) -> None:"
        )

        with buf.indent():
            out_params = [p for p in param_names if p.startswith("out_ptr")]
            in_params = [p for p in param_names if p.startswith("in_ptr")]
            ref_param = out_params[0] if out_params else param_names[0]
            ndim = self._output_ndim

            if ndim <= 1:
                # Check if the output buffer is scalar (0-dim); if so, tile
                # over an input buffer instead (full reduction case).
                out_buf_name = None
                for bname, val in self.args.output_buffers.items():
                    if not isinstance(val, RemovedArg):
                        out_buf_name = bname
                        break
                if out_buf_name is None:
                    for bname in self.args.inplace_buffers:
                        out_buf_name = bname
                        break
                out_buf_ndim = (
                    self._get_buffer_ndim(out_buf_name) if out_buf_name else 1
                )
                if out_buf_ndim == 0 and in_params:
                    ref_param = in_params[0]
                buf.writeline(f"for tile in hl.tile({ref_param}.size(0)):")
            else:
                # For reduction kernels, the output may have fewer dims than
                # ndim (e.g., var produces 1D from 2D input). Use an input
                # buffer as the tile reference when the output is too small.
                ref_buf_name = None
                for bname, val in self.args.output_buffers.items():
                    if not isinstance(val, RemovedArg):
                        ref_buf_name = bname
                        break
                if ref_buf_name is None:
                    for bname in self.args.inplace_buffers:
                        ref_buf_name = bname
                        break
                ref_ndim = (
                    self._get_buffer_ndim(ref_buf_name)
                    if ref_buf_name
                    else ndim
                )
                if ref_ndim < ndim and in_params:
                    # Find an input whose ndim matches the expected ndim.
                    # Build reverse map from inner name to outer buffer name.
                    inner_to_outer = {
                        v: k for k, v in self.args.input_buffers.items()
                    }
                    for inp in in_params:
                        inp_buf_name = inner_to_outer.get(inp)
                        if inp_buf_name is None:
                            continue
                        if self._get_buffer_ndim(inp_buf_name) == ndim:
                            ref_param = inp
                            break
                tile_vars = ", ".join(f"tile_{i}" for i in range(ndim))
                buf.writeline(
                    f"for {tile_vars} in hl.tile({ref_param}.size()):"
                )
            with buf.indent():
                for line in self.loads._lines:
                    buf.writeline(str(line))
                for line in self.compute._lines:
                    buf.writeline(str(line))
                for deferred in self.store_lines:
                    resolved = deferred()
                    if resolved is not None:
                        buf.writeline(resolved)

        return buf.getvalue()

    def call_kernel(self, name: str, node: IRNode | None = None) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.args.python_argdefs()
        call_arg_strs = [str(a) for a in call_args]

        # Initialize output buffers for scalar reductions using atomics
        if self._output_ndim == 0 and self._has_reduction():
            init_call = INIT_FOR_REDUCTION.get(self._reduction_type)
            if init_call is not None:
                for buf_name in self.args.output_buffers:
                    wrapper.writeline(f"{buf_name}{init_call}")

        # Emit reinterpret_tensor for non-contiguous input accesses
        for buf_name, (size, stride, offset) in self._input_reinterprets.items():
            inner_name = self.args.input_buffers.get(buf_name)
            if inner_name is None:
                continue
            # Find the position of this buffer in call_args
            try:
                idx = call_arg_strs.index(buf_name)
            except ValueError:
                continue
            reinterp_name = f"{buf_name}_reinterp"
            size_str = f"({', '.join(str(s) for s in size)},)"
            stride_str = f"({', '.join(str(s) for s in stride)},)"
            offset_str = str(offset)
            wrapper.writeline(
                f"{reinterp_name} = reinterpret_tensor("
                f"{buf_name}, {size_str}, {stride_str}, {offset_str})"
            )
            call_arg_strs[idx] = reinterp_name

        wrapper.writeline(f"{name}.run({', '.join(call_arg_strs)})")


def _get_helion_backend() -> str:
    device = V.graph.get_current_device_or_throw()
    if device.type in ("tpu", "xla", "cpu"):
        return "pallas"
    return config.helion_backend


class HelionScheduling(SIMDScheduling):
    kernel_type = HelionKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures) -> None:
        node_schedule = kernel_features.node_schedule
        tiling, tiling_score = self.get_tiling_and_scores(
            node_schedule,
            kernel_features.numel,
            kernel_features.reduction_numel,
            kernel_features.coalesce_analysis,
        )
        kernels = self.create_kernel_choices(
            kernel_features,
            [tiling],
            {"features": kernel_features, "tiling_scores": tiling_score},
        )

        # Determine output ndim (the number of pointwise tile dimensions).
        # Only consider final output buffers (written but not read within
        # this fused group) to avoid inflating ndim with intermediates.
        # For reduction kernels, subtract the reduction dimensions since
        # those are handled by the ':' slice, not by tile variables.
        all_writes = set()
        all_reads = set()
        for node in kernel_features.scheduler_nodes():
            for dep in node.read_writes.writes:
                all_writes.add(dep.name)
            for dep in node.read_writes.reads:
                all_reads.add(dep.name)
        final_outputs = all_writes - all_reads
        output_ndim = 0
        for name in final_outputs:
            buf = V.graph.try_get_buffer(name)
            if buf is not None:
                output_ndim = max(output_ndim, len(buf.get_size()))
        # Fallback: if no final outputs identified, use all writes
        if output_ndim == 0 and all_writes:
            for name in all_writes:
                buf = V.graph.try_get_buffer(name)
                if buf is not None:
                    output_ndim = max(output_ndim, len(buf.get_size()))
        # For reduction kernels where the output retains the same ndim as the
        # input (e.g., softmax, layer_norm), the reduction dimension is handled
        # by ':' in tile indexing, not by a tile variable. Subtract the number
        # of reduction dimensions only when the output shape hasn't already
        # accounted for the reduction (e.g., sum/var already have lower ndim).
        if kernel_features.is_reduction() and output_ndim > 1:
            from .simd import prefix_is_reduction

            n_reduction_dims = sum(
                1 for prefix in tiling if prefix_is_reduction(prefix)
            )
            max_read_ndim = 0
            for name in all_reads:
                buf = V.graph.try_get_buffer(name)
                if buf is not None:
                    max_read_ndim = max(max_read_ndim, len(buf.get_size()))
                else:
                    gi = V.graph.graph_inputs.get(name)
                    if gi is not None:
                        max_read_ndim = max(max_read_ndim, len(gi.get_size()))
            if n_reduction_dims > 0 and output_ndim >= max_read_ndim:
                output_ndim -= n_reduction_dims
        # Ensure at least 1D for non-scalar outputs
        if output_ndim > 0:
            output_ndim = max(output_ndim, 1)
        for kernel in kernels:
            kernel._output_ndim = output_ndim

        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, node_schedule, kernel)
            kernel.kernel_name = kernel_name

        (final_kernel,) = kernels
        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        final_kernel._compute_input_reinterprets()

        final_kernel.call_kernel(final_kernel.kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: Any = None,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"helion_{kernel_hash}"
        else:
            kernel_name = f"helion_{fused_name}_{kernel_hash}"
        wrapper.src_to_kernel[src_code] = kernel_name

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.helion({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
