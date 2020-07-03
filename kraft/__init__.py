from .array import (
    clip,
    error_nan,
    get_intersections_between_2_vectors,
    guess_type,
    ignore_nan_and_function_1,
    ignore_nan_and_function_2,
    is_sorted,
    log,
    normalize,
    shift_minimum,
    shuffle_slice,
)
from .cell_line import normalize_cell_line_names
from .clustering import (
    cluster,
    cluster_hierarchical_clusterings,
    get_coclustering_distance,
)
from .conda import (
    add_conda_to_path,
    get_conda_environments,
    get_conda_prefix,
    install_and_activate_conda,
    is_conda_directory_path,
)
from .context import (
    compute_pdf_and_pdf_reference_context,
    compute_vector_context,
    make_context_matrix,
    make_context_matrix_,
    plot_context,
)
from .correlation import correlate_2_vectors
from .dataframe import (
    drop_slice,
    drop_slice_greedily,
    error_axis,
    group,
    make_axis_different,
    make_axis_same,
    process_feature_x_sample,
    sample_dataframe,
    separate_type,
    summarize,
)
from .fasta import (
    get_chromosome_size_from_fasta_gz,
    get_sequence_from_fasta_or_fasta_gz,
)
from .function_heat_map import function_heat_map, function_heat_map_summary
from .gene import get_gene_symbol
from .geo import get_gse, get_key_value, parse_block
from .geometry import get_convex_hull, get_triangulation
from .gff import get_gff3_attribute, read_gff3
from .git import get_git_versions, is_in_git_repository, make_gitkeep, normalize_git_url
from .gps_map import GPSMap, read_gps_map, write_gps_map
from .grid import (
    get_grid_1ds,
    make_grid_1d,
    make_grid_1d_for_reflecting,
    make_grid_nd,
    plot_grid_nd,
    shape,
)
from .information import get_entropy, get_ic, get_jsd, get_kld, get_zd
from .internet import download, download_and_extract
from .io import read_gct, read_gmt, read_gmts, read_json, write_json
from .kernel_density import get_bandwidth, get_density
from .matrix_factorization import (
    factorize_matrix,
    factorize_matrix_by_nmf,
    plot_matrix_factorization,
    solve_ax_b,
)
from .name_biology import map_genes
from .path import get_child_paths, path
from .plot import get_color, plot_bubble_map, plot_heat_map, plot_histogram, plot_plotly
from .point import map_point, plot_node_point, pull_point
from .polar_coordinate import rescale_x_y_coordiantes_in_polar_coordiante
from .predict import train_and_classify, train_and_regress
from .probability import (
    get_posterior_probability,
    get_probability,
    plot_nomogram,
    target_posterior_probability,
)
from .sea import score_set
from .series import binarize, sample_from_each_series_value, select_extreme
from .significance import get_moe, get_p_value, get_p_values_and_q_values
from .single_cell import read_process_write_gene_x_cell
from .skew_t import (
    fit_each_dataframe_row_to_skew_t_pdf,
    fit_each_dataframe_row_to_skew_t_pdf_,
    fit_vector_to_skew_t_pdf,
)
from .string import is_version, skip_quote_and_split, standardize, title, untitle
from .support import (
    cast_builtin,
    command,
    flatten,
    get_machine,
    get_shell_environment,
    install_python_libraries,
    is_program,
    make_dict_object_i,
    make_unique,
    map_objects_to_ints,
    merge_2_dicts,
    merge_2_dicts_with_function,
    print_function_information,
)
from .tcga import select_and_group_feature_x_tcga_sample_by_sample_type
from .vcf import (
    count_gene_impacts_from_variant_dicts,
    get_variant_start_and_end_positions,
    get_variants_from_vcf_or_vcf_gz,
    get_vcf_genotype,
    get_vcf_info,
    get_vcf_info_ann,
    get_vcf_sample_format,
    make_variant_dict_from_vcf_row,
    make_variant_n_from_vcf_file_path,
    make_variant_n_from_vcf_row,
    update_variant_dict,
)
