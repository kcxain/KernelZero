from kernelzero.proposer.quality_checks import quality_issues


def test_required_operator_must_contribute_to_forward():
    module = {
        "operators": ["torch.cos"],
        "code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        unused = torch.cos(x)
        return x + 1
""",
        "raw_response": "",
    }

    assert "unused_required_result:torch.cos" in quality_issues(module)


def test_used_operator_passes_graph_checks():
    module = {
        "operators": ["torch.cos"],
        "code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        y = torch.cos(x)
        return y + x
""",
        "raw_response": "",
    }

    assert not [
        issue
        for issue in quality_issues(module)
        if issue.startswith("required_operator_not_in_forward_graph:")
        or issue.startswith("unused_required_result:")
    ]
