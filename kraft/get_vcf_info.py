def get_vcf_info(info, key):

    for info_ in info.split(sep=";"):

        if "=" in info_:

            info_key, info_value = info_.split(sep="=")

            if info_key == key:

                return info_value
