Traceback (most recent call last):
  File "/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n2_core_placed.py", line 279, in <module>
    main()
  File "/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n2_core_placed.py", line 41, in main
    my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
  File "/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n2_core_placed.py", line 74, in my_matmul
    @device(AIEDevice.npu2)
     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bcloud/torch2aie/toolchain/mlir_aie/python/aie/extras/meta.py", line 29, in builder_wrapper
    results = body_builder(*list(op_region.blocks[0].arguments))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n2_core_placed.py", line 122, in device_body
    core_tiles[row][0:n_aie_cols],
    ~~~~~~~~~~^^^^^
IndexError: list index out of range
