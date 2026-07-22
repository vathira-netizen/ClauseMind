"""The root pipeline agent.

Read CONTEXT.md first.

A ``SequentialAgent`` named ``precedent_pipeline`` whose ``sub_agents`` list
currently holds Intake & Segmentation, Precedent Retrieval, and the analysis
ParallelAgent, in that order. Later phases append the drafting LoopAgent and
Report Composer (see CONTEXT.md's pipeline diagram). Every stage
communicates exclusively through ADK session state — this file never wires
agents to call each other directly.
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from precedent.agents.analysis import analysis_stage
from precedent.agents.intake import intake_agent
from precedent.agents.retrieval_agent import retrieval_agent

root_agent = SequentialAgent(
    name="precedent_pipeline",
    description="Turns a raw contract document into a cited review report.",
    sub_agents=[intake_agent, retrieval_agent, analysis_stage],
)
