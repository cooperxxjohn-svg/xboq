"""Export layer — Excel BOQ, PDF summary, Word RFI document."""
from .excel_export import export_boq_excel
from .pdf_export import export_pdf_summary
from .word_export import export_rfi_word


class PlanExporter:
    """Stub for legacy adapter compatibility. Use export_boq_excel for real exports."""
    def export(self, payload: dict, output_path: str, **kwargs) -> str:
        return export_boq_excel(payload, output_path, **kwargs)


def export_plan(payload: dict, output_path: str, **kwargs) -> str:
    """Legacy shim — delegates to export_boq_excel."""
    return export_boq_excel(payload, output_path, **kwargs)


__all__ = [
    "export_boq_excel",
    "export_pdf_summary",
    "export_rfi_word",
    "PlanExporter",
    "export_plan",
]
