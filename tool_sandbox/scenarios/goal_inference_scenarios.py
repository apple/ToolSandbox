# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Goal inference scenarios converted from O3 Mini generated tasks.

These scenarios test the agent's ability to infer goals from natural language
descriptions and execute multi-step workflows involving various tools.
"""

import copy
import datetime
import json
from typing import Any

import polars as pl

from tool_sandbox.common.evaluation import (
    Milestone,
    SnapshotConstraint,
    addition_similarity,
    column_contains_similarity,
    snapshot_similarity,
)
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    ScenarioCategories,
)
from tool_sandbox.common.scenario import Scenario, ScenarioExtension
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.common.utils import deterministic_uuid
from tool_sandbox.scenarios.user_simulator_few_shot_examples import (
    USER_INSTRUCTION,
    named_user_simulator_few_shot_examples,
)

# mypy: disable-error-code="arg-type"

# Complete list of all tools available to AGENT (RoleType.AGENT)
ALL_AGENT_TOOLS = [
    # Contact tools
    "add_contact",
    "modify_contact",
    "remove_contact",
    "search_contacts",
    # Setting tools
    "set_low_battery_mode_status",
    "get_low_battery_mode_status",
    "set_location_service_status",
    "get_location_service_status",
    "set_cellular_service_status",
    "get_cellular_service_status",
    "set_wifi_status",
    "get_wifi_status",
    "get_current_location",
    # Messaging tools
    "send_message_with_phone_number",
    "search_messages",
    # Reminder tools
    "add_reminder",
    "modify_reminder",
    "search_reminder",
    "remove_reminder",
    # Utility tools
    "get_current_timestamp",
    "timestamp_to_datetime_info",
    "datetime_info_to_timestamp",
    "shift_timestamp",
    "timestamp_diff",
    "seconds_to_hours_minutes_seconds",
    "unit_conversion",
    "calculate_lat_lon_distance",
    "search_holiday",
    # RapidAPI search tools
    "search_lat_lon",
    "search_location_around_lat_lon",
    "search_weather_around_lat_lon",
    "search_stock",
    "convert_currency",
]


def _create_alex_dinner_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with Alex contact and dinner messages."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Alex"),
                "name": "Alex",
                "phone_number": "+15551234567",
                "relationship": "friend",
                "is_self": False,
            }
        ],
    )
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.MESSAGING,
        rows=[
            {
                "message_id": deterministic_uuid(payload="alex_dinner_msg"),
                "sender_person_id": deterministic_uuid(payload="Alex"),
                "sender_phone_number": "+15551234567",
                "recipient_person_id": deterministic_uuid(payload="Tomas Haake"),
                "recipient_phone_number": "+11233344455",
                "content": "Hey, dinner tonight at 7pm at Mario's Restaurant?",
                "creation_timestamp": (datetime.datetime.now() - datetime.timedelta(hours=2)).timestamp(),
            }
        ],
    )
    return scenario


def _create_ride_coordination_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with friend contact for ride coordination."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Sarah"),
                "name": "Sarah",
                "phone_number": "+15552345678",
                "relationship": "friend",
                "is_self": False,
            }
        ],
    )
    return scenario


def _create_alex_friend_arrival_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario for Alex friend arrival (reuse Alex contact)."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Alex"),
                "name": "Alex",
                "phone_number": "+15551234567",
                "relationship": "friend",
                "is_self": False,
            }
        ],
    )
    return scenario


def _create_charlie_dinner_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with Charlie contact for dinner plans."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Charlie"),
                "name": "Charlie",
                "phone_number": "+15559876543",
                "relationship": "friend",
                "is_self": False,
            }
        ],
    )
    return scenario


def _create_location_sharing_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with friend contact for location sharing."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Emma"),
                "name": "Emma",
                "phone_number": "+15553456789",
                "relationship": "friend",
                "is_self": False,
            }
        ],
    )
    return scenario


def _create_family_notification_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with family contact for doctor appointment."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Mom"),
                "name": "Mom",
                "phone_number": "+15554567890",
                "relationship": "family",
                "is_self": False,
            }
        ],
    )
    return scenario


def _create_emergency_contact_base(base_scenarios: dict[str, Scenario]) -> Scenario:
    """Create base scenario with sister as emergency contact."""
    scenario = copy.deepcopy(base_scenarios["base"])
    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Sister"),
                "name": "Sister",
                "phone_number": "+15555678901",
                "relationship": "family",
                "is_self": False,
            }
        ],
    )
    return scenario


def get_extensions(base_scenarios: dict[str, Scenario]) -> list[ScenarioExtension]:
    """Specify test scenario as extensions over a base scenario.

    Returns:
        A list of ScenarioExtensions for goal inference tasks
    """
    user_simulator_few_shot_examples: dict[str, list[dict[str, Any]]] = named_user_simulator_few_shot_examples()

    scenarios = []

    # Task 1: Message review + confirmation + reminder
    scenarios.append(
        ScenarioExtension(
            name="dinner_plan_review_and_confirmation",
            base_scenario=_create_alex_dinner_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Review what Alex mentioned about dinner plans in previous messages, then send him confirmation about tonight's meet-up, and set a reminder to follow up if you don't hear back by tomorrow.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I need to check something about dinner with Alex",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "search_messages",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Follow up with Alex about dinner plan",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2)],
            categories=[ScenarioCategories.MULTIPLE_TOOL_CALL, ScenarioCategories.SINGLE_USER_TURN],
        )
    )

    # Task 2: Coordinate ride with distance calculation
    scenarios.append(
        ScenarioExtension(
            name="coordinate_ride_with_distance",
            base_scenario=_create_ride_coordination_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Coordinate giving your friend a ride later today by checking distance from your place to their pickup point and sending them a message with estimated arrival time.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I need to coordinate giving someone a ride today",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "search_contacts",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "calculate_lat_lon_distance",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2)],
            categories=[
                ScenarioCategories.MULTIPLE_TOOL_CALL,
                ScenarioCategories.SINGLE_USER_TURN,
                ScenarioCategories.STATE_DEPENDENCY,
            ],
        )
    )

    # Task 3: Friend arrival - contact management + messaging + reminder
    scenarios.append(
        ScenarioExtension(
            name="friend_arrival_contact_and_reminder",
            base_scenario=_create_alex_friend_arrival_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Your friend Alex is arriving tomorrow. Make sure you have his contact information saved, send him a welcome message, and set a reminder to pick him up at the airport.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "My friend Alex is arriving tomorrow",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "search_contacts",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.CONTACT,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "name": "Alex",
                                }
                            ),
                            column_similarity_measure={"name": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Pick up Alex at airport",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2), (2, 3)],
            categories=[ScenarioCategories.MULTIPLE_TOOL_CALL, ScenarioCategories.SINGLE_USER_TURN],
        )
    )

    # Task 4: Dinner plans confirmation with Charlie
    scenarios.append(
        ScenarioExtension(
            name="dinner_plans_charlie_reminder",
            base_scenario=_create_charlie_dinner_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Confirm dinner plans with your friend Charlie and set a reminder so you don't forget the meet-up.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I need to reach out to Charlie about dinner",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "search_contacts",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Dinner with Charlie",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2)],
            categories=[ScenarioCategories.MULTIPLE_TOOL_CALL, ScenarioCategories.SINGLE_USER_TURN],
        )
    )

    # Task 5: Phone connectivity check and location sharing
    scenarios.append(
        ScenarioExtension(
            name="connectivity_check_location_share",
            base_scenario=_create_location_sharing_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Make sure your phone's connectivity is working so you can share your current location with a friend, then set a reminder to check settings later.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I want to share my location with someone but need to check my phone first",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_cellular_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_location_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Check phone settings",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 2), (1, 2), (2, 3)],
            categories=[
                ScenarioCategories.MULTIPLE_TOOL_CALL,
                ScenarioCategories.SINGLE_USER_TURN,
                ScenarioCategories.STATE_DEPENDENCY,
            ],
        )
    )

    # Task 6: Doctor's appointment reminder with location and family notification
    scenarios.append(
        ScenarioExtension(
            name="doctor_appointment_location_family_notify",
            base_scenario=_create_family_notification_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Add a reminder for your doctor's appointment tomorrow with your current location attached and notify your family about it.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I have a doctor's appointment tomorrow",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_location_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_current_location",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Doctor's appointment",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "send_message_with_phone_number",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2), (2, 3)],
            categories=[
                ScenarioCategories.MULTIPLE_TOOL_CALL,
                ScenarioCategories.SINGLE_USER_TURN,
                ScenarioCategories.STATE_DEPENDENCY,
            ],
        )
    )

    # Task 7: Doctor appointment with timestamp and emergency contact update
    scenarios.append(
        ScenarioExtension(
            name="doctor_appointment_timestamp_emergency_contact",
            base_scenario=_create_emergency_contact_base(base_scenarios),
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "You have an important doctor's appointment tomorrow at 10 AM. Set a reminder for it and make sure your emergency contact—your sister—is up-to-date in your contact list.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I have an important appointment tomorrow at 10 AM",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "datetime_info_to_timestamp",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Doctor's appointment at 10 AM",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "search_contacts",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "modify_contact",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (2, 3)],  # Reminder and contact update can be independent
            categories=[ScenarioCategories.MULTIPLE_TOOL_CALL, ScenarioCategories.SINGLE_USER_TURN],
        )
    )

    # Task 8: Dentist appointment reminder with location services
    scenarios.append(
        ScenarioExtension(
            name="dentist_appointment_location_services",
            base_scenario=base_scenarios["base"],
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "Schedule a reminder for your dentist appointment next Monday and make sure location services are active so you can get directions if needed.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I need to prepare for my dentist appointment next Monday",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_location_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Dentist appointment",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[],  # Independent tasks
            categories=[
                ScenarioCategories.MULTIPLE_TOOL_CALL,
                ScenarioCategories.SINGLE_USER_TURN,
                ScenarioCategories.STATE_DEPENDENCY,
            ],
        )
    )

    # Task 9: Office meeting with distance check and timed reminder
    scenarios.append(
        ScenarioExtension(
            name="office_meeting_distance_timed_reminder",
            base_scenario=base_scenarios["base"],
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "You have a meeting at the office tomorrow. Check your distance from it and set a reminder to leave on time.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I have a meeting at the office tomorrow",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_current_location",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "calculate_lat_lon_distance",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "datetime_info_to_timestamp",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Leave for office meeting",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 1), (1, 2), (2, 3)],
            categories=[ScenarioCategories.MULTIPLE_TOOL_CALL, ScenarioCategories.SINGLE_USER_TURN],
        )
    )

    # Task 10: Long drive preparation with multiple settings and reminder
    scenarios.append(
        ScenarioExtension(
            name="long_drive_preparation_multiple_settings",
            base_scenario=base_scenarios["base"],
            messages=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.USER,
                    "content": USER_INSTRUCTION
                    + "You're preparing for a long drive. Ensure your phone is set up for reliable navigation and connectivity. Check that cellular and location services are active, turn on WiFi for updates, and set a reminder to check traffic conditions during your trip.",
                },
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "I'm going on a long drive and need to prepare my phone",
                },
            ],
            tool_allow_list=ALL_AGENT_TOOLS,
            milestones=[
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_cellular_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "get_location_service_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                                    "recipient": RoleType.AGENT,
                                    "tool_trace": json.dumps(
                                        {
                                            "tool_name": "set_wifi_status",
                                            "arguments": {},
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ),
                        )
                    ]
                ),
                Milestone(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.REMINDER,
                            snapshot_constraint=addition_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "content": "Check traffic conditions",
                                }
                            ),
                            column_similarity_measure={"content": column_contains_similarity},
                            reference_milestone_node_index=-1,
                        ),
                    ]
                ),
            ],
            milestone_edge_list=[(0, 2), (1, 2), (2, 3)],  # Settings checks can be parallel, then WiFi, then reminder
            categories=[
                ScenarioCategories.MULTIPLE_TOOL_CALL,
                ScenarioCategories.SINGLE_USER_TURN,
                ScenarioCategories.STATE_DEPENDENCY,
            ],
        )
    )

    return scenarios


def named_goal_inference_scenarios(
    preferred_tool_backend: ToolBackend,
) -> dict[str, Scenario]:
    """Goal inference scenarios converted from O3 Mini generated tasks.

    Args:
        preferred_tool_backend: Which backend should be chosen in face of conflicting tool names.

    Returns:
        A dictionary of scenarios
    """
    from tool_sandbox.scenarios.base_scenarios import named_base_scenarios

    base_scenarios = named_base_scenarios(preferred_tool_backend)
    extensions = get_extensions(base_scenarios)

    # All scenarios in this module should be multiple tool call, single user turn by default
    for extension in extensions:
        for default_categories in [
            ScenarioCategories.MULTIPLE_TOOL_CALL,
            ScenarioCategories.SINGLE_USER_TURN,
            ScenarioCategories.GOAL_INFERENCE,
        ]:
            if default_categories not in extension.categories:
                extension.categories.append(default_categories)

    return {key: scenario for extension in extensions for key, scenario in extension.get_extended_scenario().items()}
