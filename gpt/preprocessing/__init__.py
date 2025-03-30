import typer
from .extract_abstracts import main as extract_abstracts_main

app = typer.Typer()
app.command("extract_abstracts")(extract_abstracts_main)
