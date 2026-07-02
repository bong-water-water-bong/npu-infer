#!/bin/bash
export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH
export AIETOOLS=/home/bcloud/torch2aie/toolchain/aietools
cat > /tmp/test_chess.c << 'EOF'
void test(int a, int b, int *c) { *c = a + b; }
EOF
echo "Testing Chess compiler..."
xchesscc -p me -C Release_LLVM -D__AIENGINE__ -D__AIE_ARCH__=21 -D__AIEARCH__=21 \
  -I /home/bcloud/torch2aie/toolchain/aietools/include \
  -P /home/bcloud/torch2aie/toolchain/aietools/data/aie2p/lib \
  -d -f /tmp/test_chess.c -o /tmp/test_chess.o 2>&1
echo "Exit: $?"
ls -la /tmp/test_chess.o 2>/dev/null
