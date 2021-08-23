from numpy import array

from .dictionary import summarize


def _name_feature(feature_, platform, platform_table):

    print(platform)

    platform = int(platform[3:])

    if platform in (96, 97, 570):

        label = "Gene Symbol"

        def function(
            name,
        ):

            if name != "":

                return name.split(" /// ", 1)[0]

    elif platform in (13534,):

        label = "UCSC_RefGene_Name"

        def function(
            name,
        ):

            return name.split(";", 1)[0]

    elif platform in (5175, 11532):

        label = "gene_assignment"

        def function(
            name,
        ):

            if isinstance(name, str) and name not in ("", "---"):

                return name.split(" // ", 2)[1]

    elif platform in (2004, 2005, 3718, 3720):

        label = "Associated Gene"

        def function(
            name,
        ):

            return name.split(" // ", 1)[0]

    elif platform in (10558,):

        label = "Symbol"

        function = None

    elif platform in (16686,):

        label = "GB_ACC"

        function = None

    else:

        label = None

        function = None

    for _label in platform_table.columns.values:

        if _label == label:

            print(">> {} <<".format(_label))

        else:

            print(_label)

    if label is None:

        return feature_

    else:

        name_ = platform_table.loc[:, label].values

        if callable(function):

            name_ = array(tuple(function(name) for name in name_))

        feature_to_name = dict(zip(feature_, name_))

        summarize(feature_to_name)

        return array(tuple(feature_to_name.get(feature) for feature in feature_))
