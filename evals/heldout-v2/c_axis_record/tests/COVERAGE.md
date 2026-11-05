# Heldout-v2 requirement coverage

Each spec behavior below maps to named cases in the task's test program. The C build/calls enforce C signatures;
Python `exact-public-signature` cases inspect public parameters. The runner checks target paths and fenced response
format. Every skeleton has `skeleton-fixed-lines`, which compares non-TODO template bytes.

| Task | Requirement | Named case(s) |
|---|---|---|
| `c_axis_record` | Compact axis/range parser and exact callable signature | `valid-basic`, `valid-boundaries`, C99 compile/call |
| `c_axis_record` | Exact `A<axis>,<start>,<end>` grammar; unsigned decimal; no whitespace/trailing text | `valid-basic`, `valid-leading-zeroes`, `invalid-prefix`, `invalid-empty-axis`, `invalid-negative`, `invalid-space`, `invalid-missing-comma`, `invalid-trailing` |
| `c_axis_record` | Axis 0..7; `0 <= start < end` | `valid-boundaries`, `invalid-axis-range`, `invalid-empty-range`, `invalid-reversed-range` |
| `c_axis_record` | Return 0/-1/-2/-3 for success/null/syntax/overflow-or-range | `valid-basic`, `null-text`, `null-axis`, `null-start`, `null-end`, `invalid-prefix`, `invalid-axis-range`, `invalid-int-overflow-start`, `invalid-int-overflow-end` |
| `c_axis_record` | Outputs assigned only on success | `invalid-trailing`, `invalid-axis-range`, `invalid-int-overflow-end`, `null-text`, `null-outs` |
| `c_bits3_decode` | Unpack low-bit-first unsigned 3-bit stream; value i at bit range `[3i,3i+3)`; values can cross bytes | `cross-byte-values`, `random-packed-stream`, `large-odd-count` |
| `c_bits3_decode` | Require at least ceil(3*count/8) bytes; reject arithmetic overflow | `short-source`, `random-packed-stream`, `count-arithmetic-overflow` |
| `c_bits3_decode` | Zero count succeeds without pointer dereference | `zero-count-null-pointers`, `zero-count-nonnull-pointers`, `zero-count-no-write` |
| `c_bits3_decode` | Return 0/-1; invalid input leaves dst unchanged | `short-source`, `short-source-atomic`, `null-source`, `null-source-atomic`, `null-destination`, `count-arithmetic-overflow`, `count-overflow-atomic` |
| `c_bits3_decode` | Read only required bytes; no allocation | `random-packed-stream` uses exact-sized source; `large-odd-count`; `no-heap-allocation`, caller-owned buffers |
| `c_echo_ring` | Exact `TraceRing`, init, push, get API | `init-success`, `push-success`, `get-newest`, C99 compile/call |
| `c_echo_ring` | Init requires nonnull ring/storage and positive capacity; initializes empty state | `init-null-ring`, `init-null-storage`, `init-zero-capacity`, `init-success`, `init-empty-size` |
| `c_echo_ring` | Push writes and overwrites oldest when full | `push-success`, `push-wrap`, `random-ring-push`, `large-wrap-push` |
| `c_echo_ring` | Age 0 newest; age size-1 oldest | `get-newest`, `get-oldest`, `random-ring-get`, `large-wrap-newest`, `large-wrap-oldest` |
| `c_echo_ring` | Get invalid ring/null output/unavailable age -> -1, out unchanged | `empty-get-age`, `empty-get-atomic`, `get-null-ring`, `get-null-output`, `unavailable-age`, `unavailable-age-atomic`, `invalid-null-data-get`, `invalid-zero-capacity-get`, `invalid-next-get`, `invalid-size-get` |
| `c_echo_ring` | Every operation returns 0 or -1; uses only supplied storage | `init-success`, `push-success`, `get-newest`, `push-null-ring`, `invalid-null-data-push`, `invalid-zero-capacity-push`, `invalid-next-push`, `invalid-size-push`, `no-heap-allocation`; ring storage arrays are caller-owned |
| `c_group_lse` | Stable log-sum-exp for consecutive variable-length groups | `stable-groups`, `random-variable-groups`, `largest-uneven-total` |
| `c_group_lse` | Positive lengths sum to total; x finite; output has one slot per group | `zero-group-length`, `length-sum-too-large`, `length-sum-too-small`, `nonfinite-input`, `random-variable-groups` |
| `c_group_lse` | Max subtraction used; validate nonnull pointers, positive total/groups | `stable-groups`, `null-x`, `null-lengths`, `null-output`, `zero-total`, `zero-groups`, `largest-uneven-total` |
| `c_group_lse` | Return 0/-1; all output slots unchanged on invalid input; no allocation | `zero-group-atomic`, `nonfinite-input-atomic`, `length-sum-too-large`; `no-heap-allocation`, caller-owned buffers |
| `c_rank_topk` | Top k finite values descending, ties by lower original index | `stable-tied-top-k`, `k-equals-n`, `k-one`, `random-tied-ranking`, `large-ranking` |
| `c_rank_topk` | Write exactly k index/value pairs, arrays have capacity k | `writes-exactly-k`, `writes-exactly-k-index-0`, `writes-exactly-k-index-1`, `writes-exactly-k-guard-index`, `writes-exactly-k-guard-value` |
| `c_rank_topk` | Require n>0, 1<=k<=n, nonnull pointers, all values finite | `zero-n`, `negative-n`, `zero-k`, `negative-k`, `k-too-large`, `null-values`, `null-indices`, `null-ranked`, `nan-input`, `infinite-input` |
| `c_rank_topk` | Return 0/-1; both outputs unchanged on any invalid input; no allocation | `invalid-k-atomic`, `nan-input-atomic`, `null-outputs-atomic`; output arrays allocated by caller; `no-heap-allocation` |
| `c_rms_gate` | z=x+residual; denom=sqrt(sum(z²)/n+eps); output z/denom*gate | `residual-rms-gate`, `random-rms-gate`, `largest-vector` |
| `c_rms_gate` | Nonnull pointers, n>0, finite positive eps, finite z and gate | `null-x`, `null-residual`, `null-gate`, `null-output`, `zero-n`, `negative-n`, `zero-epsilon`, `negative-epsilon`, `nan-epsilon`, `infinite-epsilon`, `nonfinite-residual-sum`, `nonfinite-gate`, `nan-z`, `nan-gate` |
| `c_rms_gate` | Return 0/-1; invalid input leaves output unchanged; disjoint buffers; float accumulation; no allocation | `epsilon-errors-atomic`, `data-errors-atomic`, `nan-gate-atomic`, `random-rms-gate`, `largest-vector`; `no-heap-allocation` |
| `c_sat8_pair` | Elementwise unsigned sum saturated at explicit cap | `saturation-ceiling`, `custom-ceiling`, `zero-ceiling`, `random-unsigned-saturation`, `largest-odd-length` |
| `c_sat8_pair` | n nonzero, all pointers nonnull, ranges supplied disjoint | `zero-length`, `null-destination`, `null-a`, `null-b`; successful cases use separate ranges |
| `c_sat8_pair` | Return 0/-1 and dst unchanged on invalid arguments; no allocation | `null-errors-atomic`, `zero-length-atomic`, `random-unsigned-saturation`; `no-heap-allocation` |
| `c_scan_checked` | Inclusive signed 32-bit prefix sums | `inclusive-scan`, `inclusive-int32-boundaries`, `random-prefix-property`, `largest-nonmultiple` |
| `c_scan_checked` | Nonnull pointers and n>0; return 0/-1/-2 for success/invalid/overflow | `null-input`, `null-output`, `zero-length`, `positive-overflow`, `negative-overflow`, `late-positive-overflow`, `late-negative-overflow` |
| `c_scan_checked` | Entire output unchanged on every error; disjoint ranges; no allocation | `null-errors-atomic`, `zero-length-atomic`, `positive-overflow-atomic`, `late-overflow-atomic`, `late-negative-atomic`; caller buffers; `no-heap-allocation` |
| `c_skeleton_clamp` | Replace only TODO bodies; preserve template bytes outside TODO | `skeleton-fixed-lines` |
| `c_skeleton_clamp` | Nonnull pointers, n>0, finite ordered bounds and finite input | `null-input`, `null-output`, `null-count`, `zero-length`, `reversed-bounds`, `nan-lower`, `infinite-upper`, `nonfinite-input`, `infinite-input` |
| `c_skeleton_clamp` | Clamp to [lo,hi], count only strictly outside values | `clamp-strict-count`, `exact-boundaries-not-counted`, `random-clamp-property`, `largest-clamp-vector` |
| `c_skeleton_clamp` | Invalid -> -1 with both outputs unchanged; valid -> 0; disjoint ranges | `null-errors-atomic`, `null-errors-count-atomic`, `zero-length-atomic`, `invalid-bounds-atomic`, `invalid-bounds-count-atomic`, `nonfinite-input-atomic`, `nonfinite-count-atomic`, `random-clamp-property` |
| `c_skeleton_pair` | Replace only TODO bodies; preserve template bytes outside TODO | `skeleton-fixed-lines` |
| `c_skeleton_pair` | Two signed decimals with one comma; optional single sign; no whitespace or trailing text | `signed-min-max`, `unsigned-decimal`, `both-signs`, `leading-zeroes`, `empty-text`, `missing-first-digit`, `missing-comma`, `missing-second-digit`, `extra-comma`, `trailing-text`, `leading-whitespace`, `embedded-whitespace`, `double-sign`, `sign-without-digit` |
| `c_skeleton_pair` | Both values fit int16; return 0/-1/-2/-3 as specified; only assign on success | `signed-min-max`, `null-text`, `null-left`, `null-right`, `positive-overflow-left`, `negative-overflow-left`, `positive-overflow-right`, `negative-overflow-right`, `random-valid-parser` (error cases assert sentinels) |
| `c_skeleton_window` | Replace only TODO bodies; preserve template bytes outside TODO | `skeleton-fixed-lines` |
| `c_skeleton_window` | Every start 0..n-width writes mean of squared window values | `adjacent-windows`, `width-one`, `width-equals-n`, `random-window-energy`, `largest-window-vector` |
| `c_skeleton_window` | Nonnull pointers, finite x, n/width positive, width<=n, exact out_n | `null-input`, `null-output`, `zero-n`, `zero-width`, `width-too-large`, `short-output`, `long-output`, `nonfinite-input`, `nan-input` |
| `c_skeleton_window` | Invalid -> -1 and output unchanged; valid -> 0; nonoverlapping buffers | `invalid-atomic`, `nonfinite-atomic`, `adjacent-windows`; separate buffers |
| `c_span_alias` | Scale dst[i]=src[i]*factor for positive n and nonnull pointers | `disjoint-scale`, `zero-factor`, `exact-in-place`, `random-disjoint-scale`, `largest-disjoint-span` |
| `c_span_alias` | Permit disjoint/exact alias; reject partial overlap | `disjoint-scale`, `exact-in-place`, `partial-overlap-forward`, `partial-overlap-reverse` |
| `c_span_alias` | Return 0/-1; leave dst unchanged on error; no allocation | `null-destination`, `null-source`, `zero-length`, `huge-byte-count`, `partial-overlap-forward-atomic`, `partial-overlap-reverse-atomic`; `no-heap-allocation` |
| `c_stable_norm` | Finite-vector softmax with max subtraction; float arithmetic; no allocation | `large-finite-logits`, `uniform-values`, `extreme-finite-gap`, `random-softmax-property`, `largest-softmax-vector`; `no-heap-allocation` |
| `c_stable_norm` | Nonnull pointers, n>0, finite inputs | `null-input`, `null-output`, `zero-n`, `negative-n`, `nan-input`, `infinite-input` |
| `c_stable_norm` | Exact alias or disjoint allowed, partial overlap invalid | `uniform-values`, `partial-overlap-forward`, `partial-overlap-reverse` |
| `c_stable_norm` | Return 0/-1 and leave output unchanged on invalid input | `null-atomic`, `invalid-size-atomic`, `nonfinite-atomic`, `partial-overlap-forward-atomic`, `partial-overlap-reverse-atomic` |
| `c_strided_proj` | Read a[r*row_stride+c*col_stride], x[c*x_stride], write contiguous row sums | `transposed-interleaved-strides`, `random-positive-strides`, `largest-transposed-grid` |
| `c_strided_proj` | Positive dimensions/strides and nonnull pointers | `null-matrix`, `null-vector`, `null-output`, `zero-rows`, `negative-rows`, `zero-cols`, `negative-cols`, `zero-row-stride`, `negative-row-stride`, `zero-col-stride`, `negative-col-stride`, `zero-vector-stride`, `negative-vector-stride` |
| `c_strided_proj` | Return 0/-1; invalid input leaves output unchanged; float accumulation; no allocation | `null-atomic`, `invalid-atomic`, `random-positive-strides`, `largest-transposed-grid`; separate buffers; `no-heap-allocation` |
| `c_u4_groupdot` | Low-first unsigned nibble q in 0..15; affine scale/bias contribution per consecutive group | `unsigned-high-nibbles-distinct-groups`, `values-eight-through-fifteen`, `odd-count-ignores-unused-high-nibble`, `random-packed-group-property`, `largest-packed-vector` |
| `c_u4_groupdot` | n and group positive, n divisible by group, all pointers nonnull | `null-packed`, `null-scales`, `null-biases`, `null-vector`, `null-output`, `zero-n`, `negative-n`, `zero-group`, `negative-group`, `nondivisible-group` |
| `c_u4_groupdot` | Return 0/-1; error leaves *out unchanged; float accumulation; no allocation | `null-atomic`, `invalid-layout-atomic`, `random-packed-group-property`, `largest-packed-vector`; `no-heap-allocation` |
| `p_affine_groups` | Exact signature; packed is rank-1 uint8; nibbles low-first unsigned 0..15 | `exact-public-signature`, `packed-rank`, `packed-dtype`, `unsigned-high-nibbles-distinct-groups`, `random-groups`, `largest-vector` |
| `p_affine_groups` | Scales/biases equal-length rank-1 real numeric arrays | `scale-rank`, `bias-rank`, `scale-bias-shape`, `scale-bool-dtype`, `scale-complex-dtype`, `bias-object-dtype`, `random-groups` |
| `p_affine_groups` | Positive integer group size not bool; unpacked length divisible; parameter count matches group count | `group-size-type`, `group-size-range`, `unpacked-not-divisible`, `group-count-mismatch`, `numpy-integer-group-size` |
| `p_affine_groups` | TypeError for dtype or packed-rank violations; ValueError for other layout violations | `packed-rank`, `packed-dtype`, `scale-rank`, `scale-bias-shape`, `group-size-type`, `group-size-range`, `unpacked-not-divisible` |
| `p_affine_groups` | Return float32 q*scale[group]+bias[group] values | `unsigned-high-nibbles-distinct-groups`, `random-groups`, `largest-vector`, `empty-packed-vector-values` |
| `p_axis_layout` | Exact signature, NumPy conversion, rank>=1 | `exact-public-signature`, `array-like-conversion`, `scalar-rank` |
| `p_axis_layout` | Integer non-bool axes; normalize negatives and validate range; TypeError/ValueError | `axis-type-bool-or-nonint`, `axis-range`, `random-axis-move-values` |
| `p_axis_layout` | NumPy moveaxis semantics; contiguous independent same-dtype copy; no input mutation | `axis-move-values`, `axis-move-contiguous`, `axis-move-independent`, `axis-move-dtype`, `axis-move-no-input-mutation`, `empty-axis-shape`, `random-axis-move-copy`, `largest-axis-move-values` |
| `p_axis_take` | Exact signature; convert array-like; scalar array ValueError | `exact-public-signature`, `scalar-array` |
| `p_axis_take` | Integer non-bool normalized axis; ValueError out of range | `axis-type`, `axis-range` |
| `p_axis_take` | Positions rank-1 signed/unsigned integer, not bool; wrong rank/dtype TypeError | `positions-rank-or-dtype`, `gather-dtype` |
| `p_axis_take` | Positions each in [0,axis length); negative values rejected with IndexError | `position-bounds-no-negative-wrap`, `random-gather-values` |
| `p_axis_take` | Return copy with axis length len(positions), preserve dtype | `gather-shape`, `gather-dtype`, `gather-copy`, `empty-positions-shape`, `random-gather-copy`, `largest-gather-values` |
| `p_block_header` | Exact signature; accept bytes/bytearray/memoryview only; no mutation | `exact-public-signature`, `bytes-input-tuple`, `bytearray-input-tuple`, `memoryview-input-tuple`, `wrong-input-type`, `bytearray-input-no-mutation`, `memoryview-input-no-mutation` |
| `p_block_header` | Little-endian 12-byte RBUF header, version 2, only flag bits 0..2 | `bad-magic`, `bad-version`, `reserved-flags`, `zero-payload`, `random-frame`, `largest-payload` |
| `p_block_header` | Exact 12+payload length; truncation/bad fields/extra or missing payload ValueError | `truncated-header`, `bad-magic`, `bad-version`, `reserved-flags`, `payload-too-short`, `payload-extra` |
| `p_block_header` | Return flags, length, sequence as Python ints | `bytes-input-python-int-types`, `random-frame`, `largest-payload` |
| `p_bytes_lru` | Exact cache API signatures; budget nonnegative Python int not bool | `exact-constructor-signature`, `exact-entry-count-signature`, `exact-bytes-property-signature`, `exact-fetch-signature`, `exact-store-signature`, `budget-nonnegative-python-int` |
| `p_bytes_lru` | Store ndarray copies as C-contiguous; key hashable; non-ndarray/hash failures TypeError | `store-strided-array`, `store-makes-contiguous-copy`, `store-isolated-from-source`, `unhashable-key`, `ndarray-required` |
| `p_bytes_lru` | Replacement removes old bytes; oversized entry uncached/False; otherwise LRU eviction then MRU insertion/True | `random-store-result`, `oversized-not-cached`, `oversized-absent`, `oversized-keeps-other-entries`, `oversized-replacement-result`, `oversized-replacement-removes-old`, `store-c-evicts-lru-b`, `random-cache-bytes` |
| `p_bytes_lru` | Fetch miss None; hit promotes and returns a copy; bytes/count report stored entries | `b-evicted`, `fetch-a-promotes`, `fetch-returns-copy`, `bytes-after-a`, `count-after-a`, `bytes-used-is-python-int`, `entry-count-is-python-int`, `random-fetch-hit`, `random-cache-entries` |
| `p_chunk_plan` | Exact signature; args Python ints not bool (TypeError otherwise) | `exact-public-signature`, `python-int-type-only` |
| `p_chunk_plan` | total>=0, chunk/alignment>0, chunk divisible by alignment (ValueError otherwise) | `range-and-alignment`, `aligned-short-tail`, `zero-total` |
| `p_chunk_plan` | Return list of (offset,size) covering [0,total) without gaps; aligned offsets, bounded sizes, only final size may be unaligned | `list-type`, `exact-coverage`, `offset`, `aligned-offset`, `size-bound`, `nonfinal-alignment`, `end`, `random-chunk-plan`, `largest-plan` |
| `p_chunk_plan` | Zero bytes gives empty list | `zero-total` |
| `p_dtype_blend` | Exact signature; broadcast a*weight+b*(1-weight); shapes must broadcast | `exact-public-signature`, `broadcasting-float32`, `random-broadcast-blend`, `broadcast-shape-mismatch`, `weight-broadcast-mismatch` |
| `p_dtype_blend` | Real integer/float inputs; reject bool/complex/object/string with TypeError | `bool-a`, `complex-a`, `object-a`, `string-a`, `bool-b`, `complex-b`, `object-b`, `string-b`, `bool-weight`, `complex-weight`, `string-weight` |
| `p_dtype_blend` | Weight finite and in [0,1], else ValueError | `invalid-weight-range-or-finiteness` |
| `p_dtype_blend` | Always float32; inputs not modified | `broadcasting-float32`, `input-a-unchanged`, `input-b-unchanged`, `weight-unchanged`, `large-inputs-unchanged`, `largest-broadcast` |
| `p_lex_order` | Exact signature; equal-length one-dimensional arrays | `exact-public-signature`, `score-rank`, `group-rank`, `length-mismatch` |
| `p_lex_order` | Real finite scores; integer non-bool groups; invalid shape/range ValueError and dtype TypeError | `score-dtype`, `nonfinite-score`, `group-dtype`, `score-rank`, `length-mismatch` |
| `p_lex_order` | Integer non-bool limit in [0,n]; return int64 first limit indices by score desc/group asc/source asc | `limit-type-bool-vs-int`, `limit-range`, `zero-limit`, `full-limit`, `score-group-source-ties`, `random-stable-ranking`, `largest-ranking`, `scores-unchanged`, `groups-unchanged` |
| `p_mask_out` | Exact signature; floating values, boolean broadcast mask, finite real scalar factor | `exact-public-signature`, `values-floating-dtype`, `mask-bool-dtype`, `factor-finite-real-scalar`, `mask-broadcast` |
| `p_mask_out` | TypeError for dtype mismatch; ValueError for invalid factor/broadcast | `values-floating-dtype`, `mask-bool-dtype`, `factor-finite-real-scalar`, `mask-broadcast` |
| `p_mask_out` | Fresh same-dtype array; selected values scaled, others copied; inputs unchanged | `broadcast-mask-values`, `float32-copy`, `independent-copy`, `no-input-mutation`, `random-masked-copy`, `largest-mask-array` |
| `p_runs_merge` | Exact signature; iterable of rank-1 sorted real arrays sharing dtype | `exact-public-signature`, `rank-or-common-real-dtype`, `unsorted-run`, `random-stable-merge` |
| `p_runs_merge` | Wrong rank/dtype/mismatched dtype TypeError; unsorted ValueError | `rank-or-common-real-dtype`, `unsorted-run` |
| `p_runs_merge` | Values preserve common dtype; origins int64; stable lower-run and within-run ties | `stable-run-ties-and-empty-run`, `values-dtype`, `origins-dtype`, `random-stable-merge`, `largest-merge` |
| `p_runs_merge` | Empty input returns empty float64/int64 arrays; outputs independent; no input mutation | `empty-iterable-contract`, `independent-output`, `inputs-unchanged`, `random-stable-merge` |
| `p_segment_pool` | Exact signature; values rank>=1; rank-1 integer ids match leading rows | `exact-public-signature`, `scalar-values`, `id-rank`, `id-count`, `id-integer-dtype` |
| `p_segment_pool` | n_segments integer non-bool >=0; ids are in [0,n); invalid shape/bounds ValueError, dtype TypeError | `segment-count-type`, `segment-count-negative`, `negative-segment-id`, `segment-id-at-upper-bound`, `nonempty-values-zero-segments`, `values-real-numeric-dtype` |
| `p_segment_pool` | Return `(n_segments,)+tail` float64 sums; empty segments zero; empty values only with empty ids; no mutation | `leading-rows-and-empty-segments`, `empty-input-valid`, `zero-segments-empty`, `empty-values-require-empty-ids`, `inputs-unchanged`, `random-segment-sum`, `largest-segment-pool` |
| `p_skeleton_calibrate` | Exact signature; only TODO body changes | `exact-public-signature`, `skeleton-fixed-lines` |
| `p_skeleton_calibrate` | Finite real raw values any shape; finite scalar offset and finite positive scalar temperature; nonscalar TypeError and invalid values/params ValueError | `stable-sigmoid-positive-negative`, `scalar-input-shape`, `invalid-raw-values`, `invalid-scalar-parameters`, `nonscalar-parameters`, `random-logistic-calibration`, `largest-score-array` |
| `p_skeleton_calibrate` | Stable sigmoid((raw-offset)/temperature), float32, same shape | `stable-sigmoid-positive-negative`, `scalar-input-shape`, `random-logistic-calibration`, `largest-score-array` |
| `p_skeleton_delta` | Exact signature; only TODO body changes | `exact-public-signature`, `skeleton-fixed-lines` |
| `p_skeleton_delta` | Nonempty rank-1 integer input; invalid shape/noninteger ValueError | `invalid-rank-empty-or-dtype`, `random-delta-encoding` |
| `p_skeleton_delta` | First little-endian signed int16; each later delta from previous original value fits int8 two's complement; return bytes; range errors ValueError | `little-endian-delta`, `single-value`, `positive-int16-endpoint`, `first-value-int16-range`, `delta-int8-range`, `random-delta-encoding`, `largest-delta-sequence` |
| `p_skeleton_diags` | Exact signature; only TODO body changes | `exact-public-signature`, `skeleton-fixed-lines` |
| `p_skeleton_diags` | Nonempty rank-2 real finite matrix; rank/empty ValueError; non-real/nonfinite TypeError | `rank-or-empty-shape`, `non-real-or-nonfinite`, `rectangular-diagonal-order`, `random-diagonal-means` |
| `p_skeleton_diags` | Return 1-D float64 means ordered from -(rows-1) through cols-1 | `rectangular-diagonal-order`, `single-row`, `single-column`, `random-diagonal-means`, `largest-diagonal-matrix` |
| `p_window_copy` | Exact signature; array-like conversion; rank>=1 | `exact-public-signature`, `scalar-rank`, `random-window-copy` |
| `p_window_copy` | Integer non-bool width/step; positive and width<=leading size; stated type/range errors | `width-step-type-bool-vs-int`, `width-step-range` |
| `p_window_copy` | Starts 0,step,2*step,... while full width fits | `leading-axis-step`, `step-larger-than-window`, `random-window-copy`, `largest-window-copy` |
| `p_window_copy` | Fresh same-dtype result with `(window_count,width)+tail` shape; output mutation isolated | `shape`, `dtype`, `copy`, `mutating-result-does-not-mutate-input`, `random-window-copy`, `largest-window-copy` |

Each Python test's randomized oracle uses fixed seed 20261105 and 256 generated cases; C test PRNGs have fixed per-task seeds and 240-256 cases. The largest fixed stress input is also compared against the same independent oracle.
