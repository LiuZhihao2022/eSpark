REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_with_Ss import EpisodeRunnerWithSs
REGISTRY["episode_with_Ss"] = EpisodeRunnerWithSs

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .whittle_cont_runner import WhittleContinuousRunner
REGISTRY["whittle_cont"] = WhittleContinuousRunner

from .parallel_runner_with_base_stock import ParallelRunnerWithBasestock
REGISTRY["parallel_with_base_stock"] = ParallelRunnerWithBasestock

from .parallel_runner_with_Ss import ParallelRunnerWithSs
REGISTRY["parallel_with_Ss"] = ParallelRunnerWithSs

from .simple_multi_echelon_episode_runner import EpisodeRunner as SimpleMultiEchelon
REGISTRY["simple_multi_echelon"] = SimpleMultiEchelon

from .parallel_runner_for_different_order_cost import ParallelRunner as ParallelForDifferentOrderCost
REGISTRY["parallel_for_different_order_cost"] = ParallelForDifferentOrderCost

from .episode_runner_for_different_order_cost import EpisodeRunner as EpisodeForDifferentOrderCost
REGISTRY["episode_for_different_order_cost"] = EpisodeForDifferentOrderCost

from .parallel_runner_downstream_fixed import ParallelRunnerWithBasestock as ParallelDownstreamFixed
REGISTRY["parallel_downstream_fixed"] = ParallelDownstreamFixed

from .episode_runner_different_cost_2_mac import EpisodeRunner4TwoMac
REGISTRY["episode_for_2_mac"] = EpisodeRunner4TwoMac

from .episode_runner_for_multi_mac import EpisodeRunner4MultiMac
REGISTRY["episode_for_multi_mac"] = EpisodeRunner4MultiMac

from .parallel_runner_different_cost_2_mac import Parallel4TwoMac
REGISTRY["parallel_for_2_mac"] = Parallel4TwoMac

from .parallel_runner_for_multi_mac import Parallel4MultiMac
REGISTRY["parallel_for_multi_mac"] = Parallel4MultiMac

from .gpt_episode_runner import GptEpisodeRunner
REGISTRY["gpt_episode"] = GptEpisodeRunner

from .gpt_episode_runner_whole_iter import GptEpisodeRunnerWholeIter
REGISTRY["gpt_episode_whole_iter"] = GptEpisodeRunnerWholeIter

from .gpt_parallel_runner_whole_iter import GptParallelRunnerWholeIter
REGISTRY["gpt_parallel_whole_iter"] = GptParallelRunnerWholeIter

from .gpt_parallel_runner import GptParallelRunner
REGISTRY["gpt_parallel"] = GptParallelRunner

from .no_gpt_parallel_runner import NoGptParallelRunner
REGISTRY["no_gpt_parallel"] = NoGptParallelRunner

from .gpt_parallel_runner_single_learner import GptParallelRunner
REGISTRY["gpt_parallel_single_learner"] = GptParallelRunner

from .gpt_test_episode_runner import GptTestEpisodeRunner
REGISTRY["gpt_test_episode"] = GptTestEpisodeRunner

from .parallel_load_mask_runner import ParallelLoadMaskRunner
REGISTRY["parallel_load_mask"] = ParallelLoadMaskRunner

from .no_gpt_parallel_runner_whole_iter import NoGptParallelRunnerWholeIter
REGISTRY["no_gpt_parallel_whole_iter"] = NoGptParallelRunnerWholeIter


