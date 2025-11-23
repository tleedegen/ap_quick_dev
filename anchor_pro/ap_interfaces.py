from typing import Union
from anchor_pro.elements.sms import SMSAnchors
from anchor_pro.elements.wood_fasteners import WoodFastener
from anchor_pro.concrete_anchors import ConcreteAnchors


SupportedWallAnchors = Union[WoodFastener, ConcreteAnchors, SMSAnchors]
SupportedBaseAnchors = Union[WoodFastener, ConcreteAnchors]

SupportedHardwareFasteners = SMSAnchors