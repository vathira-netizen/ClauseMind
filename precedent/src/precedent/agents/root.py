"""The root pipeline agent.

Read CONTEXT.md first.

A ``SequentialAgent`` named ``precedent_pipeline`` whose ``sub_agents`` list
currently holds only Intake & Segmentation. Later phases append Precedent
Retrieval, the analysis ParallelAgent, the drafting LoopAgent, and Report
Composer, in that order (see CONTEXT.md's pipeline diagram). Every stage
communicates exclusively through ADK session state — this file never wires
agents to call each other directly.
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from precedent.agents.intake import intake_agent

root_agent = SequentialAgent(
    name="precedent_pipeline",
    description="Turns a raw contract document into a cited review report.",
    sub_agents=[intake_agent],
)
