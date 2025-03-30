from typer import Typer
from gpt.categorization import app as categorization_app

app = Typer()
app.add_typer(categorization_app, name="categorization")
