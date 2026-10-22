from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["generate", "edit", "mask", "transparent", "extract"] = "generate"
    prompt: str = Field(min_length=1, max_length=12000)
    negative_prompt: str = Field(default="", max_length=6000)
    width: int = Field(default=1024, ge=256, le=4096, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=4096, multiple_of=32)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=6.0, ge=1, le=20)
    seed: int = Field(default=-1, ge=-1, le=2147483647)
    count: int = Field(default=1, ge=1, le=8)
    references: list[str] = Field(default_factory=list, max_length=10)
    mask: str | None = None
    mask_blur: int = Field(default=8, ge=0, le=64)
    preserve_unmasked: bool = True
    offload: bool = True
    vae_tiling: bool = True
    prefix_cache: bool = True

    @model_validator(mode="after")
    def check_mode(self):
        if self.mode in {"edit", "mask", "extract"} and not self.references:
            raise ValueError("Bu işlem en az bir referans görsel gerektirir.")
        if self.mode == "mask" and not self.mask:
            raise ValueError("Maske seçin veya çizin.")
        if self.mode in {"generate", "transparent"} and self.references:
            raise ValueError("Referans kullanmak için düzenleme modunu seçin.")
        if self.mode != "mask" and self.mask:
            raise ValueError("Maske yalnızca maskeleme modunda kullanılabilir.")
        return self
