import time, random
import uuid

from asyncio import Lock
from dataclasses import dataclass

from core.engine import Engine
from utils import EngineConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EngineInstance:
    engine_config: EngineConfig
    engine: Engine

    id: str
    weight: float = 0.0

    count: int = 0
    fail_count: int = 0

    last_request_time: float = 0.0
    last_success_time: float = 0.0
    last_fail_time: float = 0.0


@dataclass
class RequestInstance:
    engine_id: str
    model_name: str
    start_time: float
    end_time: float = 0


class Scheduler:
    def __init__(self):
        self.engines: dict[str, list[EngineInstance]] = {}
        self.requests: dict[str, RequestInstance] = {}

        self._request_lock = Lock()

    def add_engine(self, engine_config: EngineConfig):
        model_name = engine_config.model_name
        if model_name not in self.engines.keys():
            self.engines[model_name] = []

        engine = Engine(engine_config=engine_config)
        engine_instance = EngineInstance(
                engine_config=engine_config,
                engine=engine,
                id=str(uuid.uuid4()),
                weight=engine_config.devices.weight,
        )
        self.engines[model_name].append(engine_instance)

    async def add_request(self, request_id: str, model_name: str) -> Engine | None:
        async with self._request_lock:
            if request_id in self.requests.keys():
                logger.warning(f"Request {request_id} already exists, maybe something wrong !")
                return None
            if model_name not in self.engines.keys() or len(self.engines[model_name]) == 0:
                return None

            # 按权重选择
            weights = [engine_instance.weight for engine_instance in self.engines[model_name]]
            engine_instance: EngineInstance = random.choices(population=self.engines[model_name], weights=weights, k=1)[0]

            now = time.time()
            self.requests[request_id] = RequestInstance(engine_id=engine_instance.id, model_name=model_name, start_time=now)
            engine_instance.count += 1
            engine_instance.last_request_time = time.time()
            logger.info(f"Request {request_id} add to engine {engine_instance.id}")

            return engine_instance.engine

    async def remove_request(self, request_id: str, failed: bool) -> None:
        async with self._request_lock:
            if request_id not in self.requests.keys():
                logger.warning(f"Request {request_id} not exists, maybe something wrong !")
                return

            self.requests[request_id].end_time = time.time()
            for engine_instance in self.engines[self.requests[request_id].model_name]:
                if engine_instance.id != self.requests[request_id].engine_id:
                    continue

                if failed:
                    engine_instance.fail_count += 1
                    engine_instance.last_fail_time = self.requests[request_id].end_time
                else:
                    engine_instance.last_success_time = self.requests[request_id].end_time
                logger.info(
                        f"Request {request_id} {'success' if not failed else 'failed'}, "
                        f"cost {int(self.requests[request_id].end_time - self.requests[request_id].start_time)} seconds")
                self.requests.pop(request_id)
                return

            logger.warning(f"Request {request_id} not found in engines, engine id {self.requests[request_id].engine_id}")

    def print_status(self):
        print_str = ""
        for model_name, engine_instances in self.engines.items():
            print_str += f"\nModel {model_name} has {len(engine_instances)} engines"
            for engine_instance in engine_instances:
                print_str += f"\n\tEngine {engine_instance.id} total {engine_instance.count} requests, fail {engine_instance.fail_count} times"
        logger.info(print_str)
