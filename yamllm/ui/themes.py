import os
from pathlib import Path
import yaml
from typing import Dict, Optional, Any
from pydantic import BaseModel

class ThemeColors(BaseModel):
    primary: str = "cyan"
    secondary: str = "magenta"
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "blue"
    dim: str = "grey70"

class ThemeLayout(BaseModel):
    show_ascii_art: bool = True

class Theme(BaseModel):
    name: str
    description: str
    colors: ThemeColors = ThemeColors()
    layout: ThemeLayout = ThemeLayout()
    ascii_art: Optional[str] = None

class ThemeManager:
    def __init__(self, themes_dir: Optional[str] = None):
        if themes_dir:
            self.themes_dir = Path(themes_dir)
        else:
            self.themes_dir = Path(__file__).parent / "themes"
        
        self.themes: Dict[str, Theme] = self._load_themes()
        self.current_theme_name: str = self.get_current_theme_name()
        self.current_theme: Theme = self.themes.get(self.current_theme_name, self.themes.get("default"))

    def _load_themes(self) -> Dict[str, Theme]:
        themes = {}
        if not self.themes_dir.exists():
            return themes

        for theme_file in self.themes_dir.glob("*.yaml"):
            try:
                with open(theme_file, 'r') as f:
                    theme_data = yaml.safe_load(f)
                    theme_name = theme_file.stem
                    themes[theme_name] = Theme(**theme_data)
            except Exception:
                # Ignore invalid theme files
                pass
        
        if "default" not in themes:
            themes["default"] = Theme(name="Default", description="Default theme")

        return themes

    def get_theme(self, name: str) -> Optional[Theme]:
        return self.themes.get(name)

    def list_themes(self) -> Dict[str, Theme]:
        return self.themes

    def get_current_theme_name(self) -> str:
        # In a real app, this would read from a config file
        return "default"

    def set_theme(self, name: str):
        if name in self.themes:
            self.current_theme_name = name
            self.current_theme = self.themes[name]
            # In a real app, this would save to a config file
        else:
            raise ValueError(f"Theme '{name}' not found.")

theme_manager = ThemeManager()
