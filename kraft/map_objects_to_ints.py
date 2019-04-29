def map_objects_to_ints(iterable):

    object_int = {}

    int_object = {}

    for i, object in enumerate(sorted(set(iterable))):

        object_int[object] = i

        int_object[i] = object

    return object_int, int_object
