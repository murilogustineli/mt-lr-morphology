import typer
from .generate_answers import main as generate_answers_main

app = typer.Typer()
app.command("generate_answers")(generate_answers_main)
