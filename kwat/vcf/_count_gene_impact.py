def _count_gene_impact(vd_, ge):

    im_co = {
        "HIGH": 0,
        "MODERATE": 0,
        "LOW": 0,
        "MODIFIER": 0,
    }

    for vd in vd_:

        if vd["gene_name"] == ge:

            im_co[vd["impact"]] += 1

    return im_co
