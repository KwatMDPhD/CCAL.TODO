def get_vcf_info(io, ke):

    for ios in io.split(sep=";"):

        if "=" in ios:

            kes, vas = ios.split(sep="=")

            if kes == ke:

                return vas
