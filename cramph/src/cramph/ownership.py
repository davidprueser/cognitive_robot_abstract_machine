"""
Leaves of a statechart that another process owns, as the process ticking the
statechart holds them: it decides their life cycles, and their owner runs them and
observes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Self

from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.exceptions import UnknownProxyNodeError
from cramph.node import EndBehaviour, NodeJSONKey, StatechartNode
from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from krrood.utils import get_full_class_name

# %% what passes between the ticking process and an owner


@dataclass(kw_only=True, frozen=True)
class OwnedNodeLifeCycle:
    """
    The life cycle state the ticking process decided for a node another process owns.
    """

    owning_process: str
    """
    The name of the process owning the node.
    """

    node_id: str
    """
    The id the node has in the statechart document every process read.
    """

    run: int
    """
    How often the node started so far, so that the owner tells its runs apart.
    """

    life_cycle: LifeCycleValue
    """
    The life cycle state the node is in from now on.
    """


@dataclass(kw_only=True, frozen=True)
class OwnedNodeObservation:
    """
    What an owning process observed for its node during one run.
    """

    node_id: str
    """
    The id the node has in the statechart document every process read.
    """

    run: int
    """
    The run of the node the observation belongs to, as :attr:`OwnedNodeLifeCycle.run`
    named it.
    """

    observation: ObservationStateValue
    """
    What the owner observed.
    """


@dataclass(kw_only=True)
class OwningProcessConnection(ABC):
    """
    Hands the life cycle states of proxies on to the processes owning their nodes.
    """

    @abstractmethod
    def send_life_cycle(self, *, life_cycle: OwnedNodeLifeCycle) -> None:
        """
        Hands `life_cycle` on to the process it names, without waiting for it to
        arrive, since it is called from a tick.

        :param life_cycle: The life cycle state to hand on.
        """


# %% the ownership of one process


class ProxyJSONKey(StrEnum):
    """
    The keys of the JSON that stands for a proxy while a document is read.
    """

    PROXIED_NODE = "proxied_node"
    """
    The JSON of the node the proxy stands for, as the sender wrote it.
    """


@dataclass(kw_only=True)
class NodeOwnership(ContextExtension):
    """
    The name a process owns nodes under, and the proxies it holds of the leaves other
    processes own. In a context holding it, a statechart read from JSON holds a
    :class:`ProxyNode` for every leaf another process owns.
    """

    process_name: str
    """
    The name the nodes this process owns carry as their
    :attr:`~cramph.node.StatechartNode.owning_process`.
    """

    connection: OwningProcessConnection
    """
    Hands the life cycle states of the proxies on to their owners.
    """

    _proxies: dict[str, ProxyNode] = field(default_factory=dict, init=False, repr=False)
    """
    The proxies built in this process so far, by the id of the node they stand for.
    """

    def replace_nodes_owned_elsewhere(self, *, document: Any) -> Any:
        """
        :param document: A JSON document, or part of one, holding serialized nodes.
        :return: `document`, holding the JSON of a :class:`ProxyNode` in place of
            every node another process owns, wherever it occurs.
        """
        if isinstance(document, list):
            return [
                self.replace_nodes_owned_elsewhere(document=item) for item in document
            ]
        if not isinstance(document, dict):
            return document
        if self._is_node_owned_elsewhere(document=document):
            return {
                JSONField.TYPE: get_full_class_name(ProxyNode),
                NodeJSONKey.NODE_ID: document[NodeJSONKey.NODE_ID],
                ProxyJSONKey.PROXIED_NODE: document,
            }
        return {
            key: self.replace_nodes_owned_elsewhere(document=value)
            for key, value in document.items()
        }

    def _is_node_owned_elsewhere(self, *, document: dict[str, Any]) -> bool:
        """
        :param document: A JSON dict.
        :return: Whether `document` is a serialized node another process owns.
        """
        owning_process = document.get(NodeJSONKey.OWNING_PROCESS)
        return (
            NodeJSONKey.NODE_ID in document
            and owning_process is not None
            and owning_process != self.process_name
        )

    def add_proxy(self, *, proxy: ProxyNode) -> None:
        """
        :param proxy: A proxy that takes the observations reported for its node from
            now on.
        """
        self._proxies[proxy.node_id] = proxy

    def take_observation(self, *, observation: OwnedNodeObservation) -> None:
        """
        Hands `observation` to the proxy of its node, which observes it from its next
        tick on. May be called from any thread.

        :param observation: What an owning process observed.
        :raises UnknownProxyNodeError: If this process holds no proxy of the node.
        """
        proxy = self._proxies.get(observation.node_id)
        if proxy is None:
            raise UnknownProxyNodeError(
                process_name=self.process_name, node_id=observation.node_id
            )
        proxy.take_observation(observation=observation)


# %% the proxy


@dataclass(eq=False, repr=False, kw_only=True)
class ProxyNode(StatechartNode):
    """
    Stands in for a leaf another process owns, in a process that cannot or must not
    run it. It ends as the leaf does, sends every life cycle state it reaches to the
    owner and observes the latest observation the owner reported for its current run.

    Unlike a :class:`~cramph.node.StructureCopyNode`, it takes part in a running
    statechart; it is written to JSON as the leaf it stands for.
    """

    end_behaviour: EndBehaviour
    """
    How the leaf this proxy stands for ends on its own.
    """

    _serialized_node: dict[str, Any] | None = field(
        init=False, repr=False, default=None
    )
    """
    The JSON of the leaf this proxy stands for, as the sender wrote it.
    """

    _run: int = field(init=False, repr=False, default=0)
    """
    How often this proxy started so far.
    """

    _reported_observation: ObservationStateValue = field(
        init=False, repr=False, default=ObservationStateValue.UNKNOWN
    )
    """
    The latest observation the owner reported for the current run.
    """

    _sent_life_cycle: OwnedNodeLifeCycle | None = field(
        init=False, repr=False, default=None
    )
    """
    The life cycle state this proxy sent most recently.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        Unlike :attr:`~cramph.node.StatechartNode.required_context_extensions`, holds
        the :class:`NodeOwnership` a proxy sends its life cycle states through.
        """
        return (NodeOwnership,)

    @property
    def node_id(self) -> str:
        """
        :return: The id of the node this proxy stands for, which every process reading
            the same document knows it by.
        """
        return self._node_id

    @property
    def inherent_success_condition(self) -> Scalar:
        if self.end_behaviour.succeeds_on_observing_true:
            return self.observes_true
        return Scalar.const_false()

    @property
    def inherent_fail_condition(self) -> Scalar:
        """
        Fails as the leaf this proxy stands for does, see :attr:`end_behaviour`.
        """
        if self.end_behaviour.fails_on_observing_false:
            return self.observes_false
        return Scalar.const_false()

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :return: The JSON of the leaf this proxy stands for, so a receiver gets the leaf.
        """
        return self._serialized_node

    @classmethod
    def _create_from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Builds the proxy from the JSON of the leaf it stands for, without the class of
        that leaf.
        """
        serialized_node = data[ProxyJSONKey.PROXIED_NODE]
        proxy = DataclassJSONSerializer.from_json(
            data=serialized_node, clazz=cls, **kwargs
        )
        proxy._serialized_node = serialized_node
        return proxy

    def set_up(self) -> None:
        """
        Lets the ownership of the context hand this proxy the observations of its node.
        """
        self.context.require_extension(extension_type=NodeOwnership).add_proxy(
            proxy=self
        )

    def take_observation(self, *, observation: OwnedNodeObservation) -> None:
        """
        :param observation: What the owner observed; observed from the next tick on if
            it belongs to the current run, ignored otherwise.
        """
        if observation.run != self._run:
            return
        self._reported_observation = observation.observation

    def on_tick(self) -> ObservationStateValue:
        """
        :return: The latest observation the owner reported for the current run, unknown
            until it reported one.
        """
        return self._reported_observation

    def on_start(self) -> None:
        """
        Begins the next run, observing unknown until the owner reports, and sends it.
        """
        self._run += 1
        self._reported_observation = ObservationStateValue.UNKNOWN
        self._send_life_cycle()

    def on_pause(self) -> None:
        """
        Sends the pause to the owner.
        """
        self._send_life_cycle()

    def on_unpause(self) -> None:
        """
        Sends the end of the pause to the owner.
        """
        self._send_life_cycle()

    def on_end(self) -> None:
        """
        Sends the outcome to the owner.
        """
        self._send_life_cycle()

    def on_reset(self) -> None:
        """
        Forgets what the owner reported and sends the reset.
        """
        self._reported_observation = ObservationStateValue.UNKNOWN
        self._send_life_cycle()

    def _send_life_cycle(self) -> None:
        """
        Sends the life cycle state this proxy is in to its owner, unless it sent the
        same one of the same run before.
        """
        life_cycle = OwnedNodeLifeCycle(
            owning_process=self.owning_process,
            node_id=self.node_id,
            run=self._run,
            life_cycle=self.life_cycle_state,
        )
        if life_cycle == self._sent_life_cycle:
            return
        self._sent_life_cycle = life_cycle
        self.context.require_extension(
            extension_type=NodeOwnership
        ).connection.send_life_cycle(life_cycle=life_cycle)
