"""
XBOQ Report Generation Module

Provides professional PDF report generation from pipeline outputs.
"""

from .pdf_report import generate_bid_report, generate

__all__ = ["generate_bid_report", "generate"]
