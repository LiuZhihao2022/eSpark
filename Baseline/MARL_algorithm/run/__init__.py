from .whittle_disc_run import run as whittle_disc_run
from .whittle_cont_run import run as whittle_cont_run
from .iql_run import run as iql_run
from .ippo_run import run as ippo_run
from .ippo_with_base_stock_run import run as ippo_with_base_stock_run
from .qtran_run import run as qtran_run
from .ippo_continue_run_for_different_order_cost import run as ippo_continue_for_different_order_cost_run
from .ippo_downstream_fixed_run import run as ippo_downstream_fixed_run
from .ippo_for_2_mac_run import run as ippo_for_2_mac_run
from .ippo_for_multi_mac_run import run as ippo_for_multi_mac_run
from .gpt_as_mask_run_parallel_whole_iter import run as gpt_as_mask_run_parallel_whole_iter
from .ippo_load_mask_run import run as ippo_load_mask_run
REGISTRY = {}
REGISTRY["whittle_run"] = whittle_disc_run
REGISTRY["whittle_disc_run"] = whittle_disc_run
REGISTRY["whittle_cont_run"] = whittle_cont_run
REGISTRY["iql_run"] = iql_run
REGISTRY["ippo_run"] = ippo_run
REGISTRY["ippo_with_base_stock_run"] = ippo_with_base_stock_run
REGISTRY["qtran_run"] = qtran_run
REGISTRY["qplex_run"] = qtran_run
REGISTRY["ippo_continue_for_different_order_cost_run"] = ippo_continue_for_different_order_cost_run
REGISTRY["ippo_downstream_fixed_run"] = ippo_downstream_fixed_run
REGISTRY["ippo_for_2_mac_run"] = ippo_for_2_mac_run
REGISTRY["ippo_for_multi_mac_run"] = ippo_for_multi_mac_run
REGISTRY["gpt_as_mask_run_parallel_whole_iter"] = gpt_as_mask_run_parallel_whole_iter
REGISTRY["ippo_load_mask_run"] = ippo_load_mask_run
