Richard Math Classes
====================

Richard provides the following high level math classes (aliases in brakets):

* Vector (Array)
* Matrix (Array2)
* Kernel (Array3)

They are designed to provide good single threaded performance with nice user-friendly interfaces.

They have the following features:

* All types, regardless of their dimensionality, store their data as a flat array.
* All types are convertable into one-another without copying data. This is achieved without class inheritence to avoid the overhead of virtual function calls.
* From an instance you can obtain another "shallow" instance or "slice" that points to the parent's data. Modifications to the child instance are reflected in the parent.
* All types define move constructors and move assignment operators.

Usage examples
--------------

# TODO

```

```

Refer to the unit tests for more examples.

Shallow instances
-----------------

# TODO

Assignment
----------

Assigning to a shallow instance performs a copy operation.

```
    Vector A({ 1, 2, 3, 4, 5 });
    VectorPtr pB = Vector::subvector(1, 3);
    Vector& B = *pB;

    // B is now [ 2, 3, 4 ]

    B = Vector({ 1, 1, 1 }) * 9;

    // B is now [ 9, 9, 9 ]
    // A is now [ 1, 9, 9, 9, 5 ]
```

An exception is thrown if the RHS vector's size doesn't match the LHS vector's size.

```
    B = Vector({ 1, 1, 1, 1 }) * 9; // Error! B is a shallow vector of size 3
```

### Common pitfalls

Reassigning an array will invalidate any outstanding shallow subarrays.

```
    Vector A({ 1, 2, 3, 4, 5 });
    ConstVectorPtr pB = Vector::subvector(1, 3);
    const Vector& B = *pB;

    A = Vector({ 6, 7, 8 }); // B is now invalid!

    std::cout << B[1] << std::endl; // Undefined behaviour
```

