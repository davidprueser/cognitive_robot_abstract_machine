"""
Blackboard identifier constants for RoboKudo.

This module provides a set of constants used to identify and access data stored in the
behavior tree's blackboard. These identifiers are used consistently across the codebase
to ensure proper data access and communication between components.
"""

from enum import StrEnum


class BBIdentifier(StrEnum):
    """
    Constants for accessing data on the behavior tree's blackboard.

    This class defines string constants that serve as keys for storing and
    retrieving data from the blackboard. The constants are used primarily for:

    * Action server communication
    * Query handling
    * Exception management

    The constants ensure consistent access to blackboard data across the system.
    """

    QUERY_SERVER = "query_server"
    """
    Action server instance for handling queries.
    """

    QUERY_SERVER_IN_PIPELINE = "query_server_in_pipeline"
    """
    Indicate that a node in the pipeline actually requires the Action Server.
    """

    QUERY_ANSWER = "query_answer"
    """
    Result message instance from the action server (typically
    `robokudo_msgs.msg.QueryResult`)
    """

    QUERY_FEEDBACK = "query_feedback"
    """
    Feedback message for the action server.
    """

    QUERY_PREEMPT_REQUESTED = "query_preempt_requested"
    """
    Flag indicating if preemption was requested.
    """

    QUERY_PREEMPT_ACK = "query_preempt_ack"
    """
    Flag indicating if preemption was acknowledged.
    """

    BLACKBOARD_EXCEPTION_NAME = "exception"
    """
    Name for storing exceptions on the blackboard.
    """

    BELIEF_STATE = "belief_state"
    """
    The :class:`~robokudo.belief_state.BeliefState` of this process.
    """
