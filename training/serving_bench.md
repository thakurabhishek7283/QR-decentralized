Closed-loop benchmark (`python -m sigil.inference.bench`): N clients each send the next
request as soon as the previous answer arrives, for 5 s after a 1 s warm-up. Each request
is the full `InferenceService.score` path on a real simulated patch (classical
correlation, `make_input`, ONNX Runtime). 4 vCPU container, ORT intra-op threads left at
the default, single process, other light jobs sharing the machine, so treat differences
under ~15% as noise. "unbatched" is `max_batch_size=1`; "batched" is the default
`max_batch_size=32, max_delay_ms=4`.

| concurrency | mode | req/s | speed-up | p50 ms | p95 ms | p99 ms | mean batch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | unbatched | 420.2 | 1.00x | 2.2 | 3.2 | 4.4 | 1.0 |
| 1 | batched | 397.6 | 0.95x | 2.3 | 3.6 | 5.5 | 1.0 |
| 8 | unbatched | 362.6 | 1.00x | 20.1 | 37.6 | 52.3 | 1.0 |
| 8 | batched | 580.4 | 1.60x | 13.4 | 18.2 | 23.0 | 8.0 |
| 32 | unbatched | 432.8 | 1.00x | 71.1 | 92.1 | 190.8 | 1.0 |
| 32 | batched | 559.2 | 1.29x | 55.8 | 68.7 | 79.8 | 32.0 |

How to read it:

* The CNN itself costs about 0.3 ms per patch (0.2 ms per patch in a batch of 8-32), while
  pre-processing on the event loop (Pearson correlation plus `make_input`) costs about
  1 ms. A single Python process therefore tops out near 500-600 req/s whatever the
  batcher does, and batching can only remove the small model share, which is why the
  gain is 1.3-1.6x rather than an order of magnitude. Scaling further means more
  processes, not a bigger batch.
* With one client, batching is neutral (0.95x is within noise). The service tells the
  batcher how many requests are in flight, so a lone request is dispatched immediately
  instead of idling for `max_delay_ms`.
* Two defects were found and fixed while producing these numbers. ONNX Runtime's
  intra-op workers busy-wait between calls; on four shared cores that starved the
  event loop and the batcher thread, and switching it off
  (`session.intra_op.allow_spinning=0`) lifted unbatched throughput from about 125 to
  about 450 req/s with p50 falling from 5.5 ms to 2.2 ms. Before the in-flight hint, a
  lone request in batched mode always waited out the 4 ms delay (0.3-0.6x of unbatched
  at concurrency 1).
* Intra-op thread counts of 1, 2 and 4 gave the same throughput within noise once
  spinning was off, so the default (ORT chooses) is kept.
