def map_objects_to_ints(iterable):

    object_int = {}

    int_object = {}

    for i, object_ in enumerate(sorted(set(iterable))):

        object_int[object_] = i

        int_object[i] = object_

    return object_int, int_object
