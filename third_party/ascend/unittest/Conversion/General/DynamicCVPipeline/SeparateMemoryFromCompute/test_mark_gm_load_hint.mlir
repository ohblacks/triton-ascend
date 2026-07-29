// RUN: triton-opt --mark-gm-load %s | FileCheck %s

// ============================================================================
// Test 1 (hint force-on depth=2): annotation.mark on the tensor produced by
// bufferization.to_tensor after memref.copy, matching the real pre-bufferization
// pipeline order.
// ============================================================================

// CHECK-LABEL: func.func @hint_force_on
func.func @hint_force_on(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf16>
      // CHECK: memref.alloc() : memref<128xf16>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
      %tensor = bufferization.to_tensor %alloc restrict writable : memref<128xf16> to tensor<128xf16>
      annotation.mark %tensor {gm_load = 2 : i32} : tensor<128xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 2 (hint force-off depth=0): gm_load = 0, expect NO
// hivm.multi_buffer marking.
// ============================================================================

// CHECK-LABEL: func.func @hint_force_off
func.func @hint_force_off(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
      %tensor = bufferization.to_tensor %alloc restrict writable : memref<128xf16> to tensor<128xf16>
      annotation.mark %tensor {gm_load = 0 : i32} : tensor<128xf16>
      // CHECK-NOT: hivm.multi_buffer
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 3 (hint force-on depth=3): gm_load = 3.
// ============================================================================

// CHECK-LABEL: func.func @hint_force_on_depth3
func.func @hint_force_on_depth3(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf16>
      // CHECK: memref.alloc() : memref<128xf16>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 3 : i32} : memref<128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
      %tensor = bufferization.to_tensor %alloc restrict writable : memref<128xf16> to tensor<128xf16>
      annotation.mark %tensor {gm_load = 3 : i32} : tensor<128xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 4 (no hint, auto): no compile hint, default bufferCount=1 means no
// multi_buffer annotation is emitted.
// ============================================================================

// CHECK-LABEL: func.func @hint_none_auto
func.func @hint_none_auto(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf16>
      // CHECK: memref.alloc() : memref<128xf16>
      // CHECK-NEXT: memref.copy
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 5 (hint force-off depth=1): gm_load = 1 is also force-off.
// ============================================================================

// CHECK-LABEL: func.func @hint_force_off_depth1
func.func @hint_force_off_depth1(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
      %tensor = bufferization.to_tensor %alloc restrict writable : memref<128xf16> to tensor<128xf16>
      annotation.mark %tensor {gm_load = 1 : i32} : tensor<128xf16>
      // CHECK-NOT: hivm.multi_buffer
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
