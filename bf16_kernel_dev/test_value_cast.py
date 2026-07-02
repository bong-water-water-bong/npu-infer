#!/usr/bin/env python3
"""Test whether ScalarValue instances pass nanobind's Value type check."""
import sys, os

sys.path.insert(0, '/home/bcloud/mlir-aie/build/python')
sys.path.insert(1, '/home/bcloud/torch2aie/toolchain/xrt/python')
os.environ['LD_LIBRARY_PATH'] = '/home/bcloud/mlir-aie/build/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.insert(0, '/home/bcloud/mlir-aie/build/python')
from aie.ir import Context, Module, InsertionPoint, Location
from aie.dialects import scf
from aie.dialects import arith as aie_arith
from aie.dialects import func
from aie.dialects._ods_common import _cext

# Set up a module
ctx = Context()
mod = Module.create(Location.unknown(ctx))
with Context(), Location.unknown():
    # Get a Value first
    i32_type = _cext.ir.IntegerType.get_signless(32)
    idx_type = _cext.ir.IndexType.get()
    
    f = func.FuncOp("test", ([], []))
    entry_block = f.add_entry_block()
    with InsertionPoint(entry_block):
        # Create a raw Value
        c0 = aie_arith.ConstantOp(idx_type, 0)
        c1 = aie_arith.ConstantOp(idx_type, 1)
        c10 = aie_arith.ConstantOp(idx_type, 10)
        
        v0 = c0.result  # This is a raw Value
        v1 = c1.result
        v10 = c10.result
        
        print("=== Test 1: Raw Values to ForOp ===")
        print(f"v0 type: {type(v0)}")
        print(f"v0 instanceof Value: {isinstance(v0, _cext.ir.Value)}")
        
        try:
            loop = scf.ForOp([], v0, v10, v1, [])
            print("Raw Values -> ForOp: OK")
        except Exception as e:
            print(f"Raw Values -> ForOp: FAILED: {e}")
        
        # Now test with ArithValue / ScalarValue
        print("\n=== Test 2: Import ScalarValue ===")
        from aie.extras.dialects.arith import ScalarValue, ArithValue
        
        sv0 = ScalarValue(v0)
        sv1 = ScalarValue(v1)
        sv10 = ScalarValue(v10)
        
        print(f"sv0 type: {type(sv0)}")
        print(f"sv0 isinstance Value: {isinstance(sv0, _cext.ir.Value)}")
        print(f"sv0 isinstance ArithValue: {isinstance(sv0, ArithValue)}")
        print(f"sv0 isinstance ScalarValue: {isinstance(sv0, ScalarValue)}")
        
        # Try to pass ScalarValue to ForOp
        print("\n=== Test 3: ScalarValues to ForOp ===")
        try:
            loop2 = scf.ForOp([], sv0, sv10, sv1, [])
            print("ScalarValues -> ForOp: OK")
        except Exception as e:
            print(f"ScalarValues -> ForOp: FAILED: {e}")
        
        # Test with individual ScalarValue as operands
        print("\n=== Test 4: Try casting ScalarValue via C++ ===")
        try:
            # Test if ScalarValue works with arith ops
            add = aie_arith.AddIOp(sv0, sv1)
            print(f"Arith.AddIOp(ScalarValue): OK -> {type(add.result)}")
        except Exception as e:
            print(f"Arith.AddIOp(ScalarValue): FAILED: {e}")
        
        # Test wrapping in Value()
        print("\n=== Test 5: Wrap ScalarValue in Value() ===")
        vv0 = _cext.ir.Value(sv0)
        print(f"Value(ScalarValue) type: {type(vv0)}")
        try:
            loop3 = scf.ForOp([], vv0, v10, v1, [])
            print("Value(ScalarValue) -> ForOp: OK")
        except Exception as e:
            print(f"Value(ScalarValue) -> ForOp: FAILED: {e}")
        
        # Test with str() conversion  
        print(f"\n=== Test 6: str(ScalarValue) ===")
        print(f"sv0: {sv0}")
        
        # Clean up
        print("\n=== Done ===")

print("Module:", str(mod))
