# This file's code is copied from dlora-artifact
# Original project repository address: https://github.com/LLMServe/dLoRA-artifact
# Original author(s): Wu, B., Zhu, R., Zhang, Z., Sun, P., Liu, X., & Jin, X. (2024).
# Original license: Apache License 2.0
# 
# Copyright 2024 Wu, B., Zhu, R., Zhang, Z., Sun, P., Liu, X., & Jin, X. .
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import argparse
import pandas as pd
import numpy as np
import heapq

from typing import Iterable, Dict, List, Tuple, Iterator

class ArrivalProcess(ABC):
    @abstractmethod
    def rate(self) -> float:
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self) -> float:
        """Return the coefficient of variation of the gap between
        the prompts."""
        raise NotImplementedError()

    @abstractmethod
    def get_iterator(self, start: float, duration: float,
                          seed: int = 0) -> Iterator[float]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"rate={self.rate()}, "
                f"cv={self.cv()})")

    def params(self) -> Tuple[str, str]:
        return self.rate(), self.cv()

class GammaProcess(ArrivalProcess):
    """Gamma arrival process."""
    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.
        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self) -> float:
        return self.rate_

    def cv(self) -> float:
        return self.cv_

    def get_iterator(self, start: float, duration: float, seed: int = 0) -> Iterator[float]:
        np.random.seed(seed)

        batch_size = max(int(self.rate_ * duration * 1.2), 1)
        # 根据shape和scale获得一个服从gamma分布的时间间隔列表，列表长度为batch_size
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        # 当前选取的间隔的索引
        pt = 0
        # 当前时间
        cur = start + intervals[0]
        # 结束时间
        end = start + duration
        while True:
            yield cur

            pt += 1
            if pt >= batch_size: # 如果当前选取的间隔索引超过了batch_size，则重新生成一个间隔列表
                intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                pt = 0

            # 当前时间加上当前选取的间隔，得到下一个时间节点
            cur += intervals[pt]
    
class PoissonProcess(GammaProcess):
    """Poisson arrival process."""

    def __init__(self, arrival_rate: float):
        """Initialize a Poisson arrival process.
        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)

def load_trace(trace_name, trace_dir, duration_list, end_d, end_h, end_m, start_d=0, start_h=0, start_m=0, need_sort=False):
    if trace_name == "azure_v1":
        usecols = ['HashOwner', 'HashApp', 'HashFunction'] + duration_list
        trace_md = pd.read_csv(trace_dir, usecols=usecols)
        trace_md['InvocationSum'] = 0
        for duration in duration_list:
            trace_md['InvocationSum'] += trace_md[duration]
        trace_md = trace_md[trace_md['InvocationSum'] > 0]
        function_name = trace_md.groupby(['HashOwner', 'HashApp', 'HashFunction']).size().reset_index(name='FunctionName')
        function_name['FunctionName'] = np.arange(0, len(function_name))
        assert len(trace_md) == len(function_name)
        trace_md = trace_md.merge(function_name, on=['HashOwner', 'HashApp', 'HashFunction'], how='inner')   
        if need_sort:
            sorted_md = trace_md.sort_values(by=['InvocationSum'], ascending=False)
            names = sorted_md['FunctionName'].to_numpy()
        else:
            names = function_name['FunctionName'].to_numpy() 
        return trace_md, names # 
    elif trace_name == "azure_v2":
        usecols = ['app', 'func', 'end_timestamp']
        trace_md = pd.read_csv(trace_dir, usecols=usecols)
        function_name = trace_md.groupby(['app', 'func']).size().reset_index(name='name')
        function_name['name'] = np.arange(0, len(function_name))
        trace_md = trace_md.merge(function_name, on=['app', 'func'], how='inner')
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        trace_md = trace_md[(trace_md['end_timestamp'] >= start_timestamp_seconds) & (trace_md['end_timestamp'] < end_timestamp_seconds)]
        if need_sort:
            names, counts = np.unique(trace_md['name'], return_counts=True)
            sorted_indices = np.argsort(-counts)
            names = names[sorted_indices]
        else:
            names = function_name['name'].to_numpy() 

        return trace_md, names

    
def generate_from_iterators(iterators: Dict[int, Iterator], num_reqs: int):
    heap: List[Tuple[float, int]] = []

    # 遍历每个模型的迭代器，获取第一个值，并将其加入到堆中
    # heapq是一个优先队列，堆顶元素是最小值
    # heapq.heappush(heap, (val, model_id))，将(val, model_id)加入到堆中
    for model_id, iter in iterators.items():    
        heapq.heappush(heap, (next(iter), model_id))
    
    result: List[Tuple[float, int]] = []
    num_generated = 0

    # 生成num_reqs个请求
    while num_generated < num_reqs:
        # 从堆中弹出最小值，并将其加入到结果中
        val, model_id = heapq.heappop(heap)
        result.append((val, model_id))
        num_generated += 1

        # 获取当前模型的下一个值（到达时间），并将其加入到堆中
        next_val = next(iterators[model_id])
        heapq.heappush(heap, (next_val, model_id))

    return result

def timestr_to_dhm(time_str):
    dhm = time_str.split(sep=".")
    if len(dhm) != 3:
        raise RuntimeError("Wrong format for `start_time`.")
    day = int(dhm[0])
    hour = int(dhm[1])
    min = int(dhm[2])
    return day, hour, min

class Trace:
    def __init__(
        self, 
        trace_name: str, 
        trace_dir: str,
        start_time: str, 
        end_time: str,
        need_sort: bool = False,
    ):
        self.trace_name = trace_name
        self.trace_dir = trace_dir
        self.start_d, self.start_h, self.start_m = timestr_to_dhm(start_time)
        self.end_d, self.end_h, self.end_m = timestr_to_dhm(end_time)
        self.start_mnt = self.start_d * 24 * 60 + self.start_h * 60 + self.start_m
        self.end_mnt = self.start_d * 24 * 60 + self.end_h * 60 + self.end_m
        self.duration = self.end_mnt - self.start_mnt
        if trace_name == "azure_v1":
            # now only support for one day in maf1
            # and must assert trace_dir corresponds to the day
            assert self.end_d == self.start_d
            if self.start_d < 9:
                trace_dir += f"invocations_per_function_md.anon.d0{self.start_d+1}.csv"
            else:
                trace_dir += f"invocations_per_function_md.anon.d{self.start_d+1}.csv"
            self.duration_list = [str(i) for i in range(self.start_mnt+1, self.end_mnt+1)]
            self.function_histogram, self.function_names = load_trace(trace_name, trace_dir, self.duration_list, self.end_d, self.end_h, self.end_m, self.start_d, self.start_h, self.start_m, need_sort=need_sort)
            self.num_req = len(self.function_histogram)
        elif trace_name == "azure_v2":
            trace_dir += 'AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt'
            self.function_arrivals, self.function_names = load_trace(trace_name, trace_dir, [], self.end_d, self.end_h, self.end_m, self.start_d, self.start_h, self.start_m, need_sort=need_sort)
            self.num_req = len(self.function_arrivals)
        else:
            raise NotImplementedError(f"trace_name {trace_name} not supported")

    # 如果map_stride=1, num_lora_models=8, function_names=[0,1,2,3,4,5,6,7], 那么映射关系为：
    # {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7]}
    # 如果map_stride=2, num_lora_models=8, function_names=[0,1,2,3,4,5,6,7], 那么映射关系为：
    # {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
    # 如果map_stride=3, num_lora_models=8, function_names=[0,1,2,3,4,5,6,7], 那么映射关系为：
    # {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7]}
    # 如果map_stride=4, num_lora_models=8, function_names=[0,1,2,3,4,5,6,7], 那么映射关系为：
    # {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    def map_model(self, num_lora_models: int, function_names: Iterable[int], map_stride: int = 1):
        mapping: Dict[int, List[int]] = {}
        num_functions = len(function_names)
        assert num_functions >= num_lora_models, f"#function {num_functions} < #models {num_lora_models}"
        rest_stride = map_stride
        model_id = 0
        for _, func in enumerate(function_names):
            if model_id not in mapping:
                mapping[model_id] = [func]
            else:
                mapping[model_id].append(func)
            rest_stride -= 1
            if rest_stride == 0: # 每隔rest_stride个function就换一个model
                rest_stride = map_stride
                model_id = (model_id + 1) % num_lora_models
        return mapping
    
    def replay_to_workload(self, num_lora_models: int, num_reqs: int, arrival_distribution: str="gamma", 
                           interval_minutes: int=5, tot_rate: float = 4.0, cv: float = 1.0, map_stride: int = 1) -> List[Tuple[int, float]]:
        
        # num_reqs = min(num_reqs, self.num_req)
        model_mapping = self.map_model(num_lora_models, self.function_names, map_stride)
        model_histogram: Dict[int, np.array] = {}
        model_arrivals: Dict[int, np.array] = {}
        if self.trace_name == "azure_v1":
            for model, functions in model_mapping.items():
                model_cnts = np.zeros((self.duration,))
                for func in functions:
                    func_cnts = self.function_histogram.loc[self.function_histogram['FunctionName']==func].iloc[0][self.duration_list].to_numpy()
                    model_cnts = np.add(model_cnts, func_cnts)
                model_histogram[model] = model_cnts
        elif self.trace_name == "azure_v2":
            func_mapping = {}
            for model, functions in model_mapping.items():
                for func in functions:
                    func_mapping[func] = model
            # 统计每个函数的到达时间
            for func, model in func_mapping.items():
                func_arrivals = self.function_arrivals[self.function_arrivals['name'] == func]['end_timestamp'].to_numpy()
                if model not in model_arrivals:
                    model_arrivals[model] = func_arrivals
                else:
                    model_arrivals[model] = np.concatenate((model_arrivals[model], func_arrivals))

            for model, arrivals in model_arrivals.items():
                model_arrivals[model] = np.sort(arrivals)            

        dataset: Dict[int, np.array] = {}
        num_intervals = (self.duration + interval_minutes - 1) // interval_minutes
        if self.trace_name == "azure_v1":
            for model, model_cnts in model_histogram.items():
                assert len(model_cnts) == self.duration
                accumulated = np.zeros((num_intervals,))
                for i in range(accumulated.size):
                    start = i * interval_minutes
                    end = (i + 1) * interval_minutes if (i + 1) * interval_minutes <= self.duration else self.duration
                    accumulated[i] = np.sum(model_cnts[start:end])
                dataset[model] = accumulated
        else:
            intervals = np.arange(self.start_mnt, self.end_mnt, interval_minutes)
            if intervals[-1] != self.end_mnt:
                intervals = np.append(intervals, self.end_mnt)
            # 统计每个模型在每个时间段内的到达次数
            for m in model_arrivals: 
                arrivals = model_arrivals[m]
                interval_dataset = []
                # 遍历每一个时间段
                for i in range(intervals.size - 1):
                    tmp = arrivals[arrivals >= intervals[i] * 60]
                    tmp = tmp[tmp < intervals[i+1] * 60]
                    # 将这个时间段内的到达次数加入到interval_dataset中
                    interval_dataset.append(len(tmp))
                # dataset[m]中保存了第m个lora model在每个时间段内的到达次数
                dataset[m] = np.array(interval_dataset)

        distributions = self.estimate_parameters_with_histogram(dataset, arrival_distribution, tot_rate, cv)

        # 将所有请求次数均匀分配到每个时间段
        num_reqs_per_interval = num_reqs // num_intervals

        replay_trace: List[Tuple[float, int]] = []
        start = 0
        # 遍历每个时间段
        for i in range(num_intervals):
            iterator_list: Dict[int, Iterator] = {}
            # 遍历每个模型在当前时间段的到达模式
            for model, arrival_process in distributions.items():
                if arrival_process[i] is None:
                    continue
                iterator_list[model] = arrival_process[i].get_iterator(start, interval_minutes)
            num_reqs_i = num_reqs_per_interval
            # 如果当前时间段的请求数不能被num_intervals整除，则将剩余的请求数分配到前num_reqs % num_intervals个时间段
            if i < num_reqs % num_intervals:
                num_reqs_i += 1
            # interval_trace中保存了当前时间段内的所有请求的到达时间和对应模型id
            interval_trace = generate_from_iterators(iterator_list, num_reqs_i)
            replay_trace.extend(interval_trace)
            # 将当前时间段的结束时间作为下一个时间段的开始时间
            start, _ = replay_trace[-1]

        workload: List[Tuple[int, float]] = []
        models_cnt = [0] * num_lora_models
        pre_arrival = 0
        for arrival, model in replay_trace:
            models_cnt[model] += 1
            assert pre_arrival <= arrival
            # workload中保存每个请求的模型id以及与上一个请求的时间间隔
            workload.append((model, arrival - pre_arrival))
            pre_arrival = arrival

        print(models_cnt)
        # print(workload)
        print("last arrival:", arrival)

        return workload

    def estimate_parameters_with_histogram(self,
                                           dataset,
                                           arrival_distribution="exponential",
                                           tot_rate=4.0,
                                           cv=1.0) -> Dict[int, List[ArrivalProcess]]:
        if arrival_distribution not in ["exponential", "gamma"]:
            raise NotImplementedError(f"We can only use histogram data for exponential or gamma distribution, "
                                      f"got {arrival_distribution}")
        distributions: Dict[int, List[ArrivalProcess]] = {}
        sum_hist = None
        # 在各个时间段的所有模型的调用次数之和
        for _, histogram in dataset.items():
            if sum_hist is None:
                sum_hist = list(histogram)
            else:
                sum_hist += histogram

        for model, histogram in dataset.items():
            distributions[model] = []
            # id：第id个时间间隔，h：第id个时间间隔内的调用次数
            for id, h in enumerate(histogram):
                if h == 0: # 如果当前时间段内没有调用次数，则不需要生成到达率
                    distributions[model].append(None)
                else:
                    rate_ratio = h / sum_hist[id]  # 当前模型在当前时间段的调用次数占所有模型在当前时间段的调用次数的比例
                    arrival_rate = rate_ratio * tot_rate  # 当前模型在当前时间段的到达率
                    if arrival_distribution == "exponential":
                        distributions[model].append(PoissonProcess(arrival_rate))
                    else:
                        distributions[model].append(GammaProcess(arrival_rate, cv))
        return distributions



