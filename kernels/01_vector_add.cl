__kernel void vector_add(global const int* a,
                         global const int* b,
                         global int* c) {
    const unsigned int idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
}

__kernel void vector_add_return(global const int* a,
                                global const int* b,
                                global int* c,
                                unsigned int size) {
    const unsigned int idx = get_global_id(0);
    if (idx >= size)
        return;
    c[idx] = a[idx] + b[idx];
}
