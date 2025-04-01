import typer
from .generate_answers import main as generate_answers_main
from .categorize import main as categorize_main

app = typer.Typer()
app.command("generate_answers")(generate_answers_main)
app.command("categorize")(categorize_main)
