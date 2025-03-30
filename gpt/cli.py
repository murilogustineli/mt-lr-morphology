from typer import Typer
from gpt.categorization import app as categorization_app
from gpt.preprocessing import app as preprocessing_app

app = Typer()
app.add_typer(categorization_app, name="categorization")
app.add_typer(preprocessing_app, name="preprocessing")
