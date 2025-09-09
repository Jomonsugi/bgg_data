"""
Typed Event classes for the LlamaIndex Workflow steps.
"""

from llama_index.core.workflow import Event


class ProcessNextGameEvent(Event):
    pass


class PlanStrategiesEvent(Event):
    pass


class TryBGGOfficialLinkEvent(Event):
    pass


class TryTavilyPdfSearchEvent(Event):
    pass


class TryWebsiteProbeEvent(Event):
    pass


class TryComprehensiveSeleniumEvent(Event):
    pass


class TryDirectPdfCandidateEvent(Event):
    pass


class TryDirectPdfDownloadEvent(Event):
    pass


class StrategyNextEvent(Event):
    pass


