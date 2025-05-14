import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..train.helper_functions import load_dataset_from_gcp
from ..worker import celery_app


@celery_app.task(bind=True)
def plot(self, file_url, plot_form, plot_type):
    try:
        df = load_dataset_from_gcp(file_url)
        if isinstance(df, dict) and "message" in df:
            return {"message": df["message"]}

        plt.figure(figsize=(6, 6))
        
        if plot_type == "line":
            plt.plot(df[plot_form["x"]], df[plot_form["y"]])
        elif plot_type == "bar":
            agg_series = df.groupby(plot_form["x"])[plot_form["y"]].agg(plot_form["aggregation"])
            agg_series = agg_series.reset_index()
            plt.bar(agg_series[plot_form["x"]], agg_series[plot_form["y"]])
        elif plot_type == "scatter":
            plt.scatter(df[plot_form["x"]], df[plot_form["y"]])
        elif plot_type == "box":
            if plot_form.get("hue"):
                sns.boxplot(data=df, x=plot_form["x"], y=plot_form["y"], hue=plot_form["hue"])
            else:
                sns.boxplot(data=df, x=plot_form["x"], y=plot_form["y"])
        elif plot_type == "violin":
            if plot_form.get("hue"):
                sns.violinplot(data=df, x=plot_form["x"], y=plot_form["y"], hue=plot_form["hue"])
            else:
                sns.violinplot(data=df, x=plot_form["x"], y=plot_form["y"])
        elif plot_type == "hist":
            plt.hist(df[plot_form["x"]], bins=20)
        elif plot_type == "kde":
            if plot_form.get("hue"):
                sns.kdeplot(data=df, x=plot_form["x"], hue=plot_form["hue"])
            else:
                sns.kdeplot(data=df, x=plot_form["x"])
        elif plot_type == "pie":
            agg_series = df.groupby(plot_form["x"])[plot_form["y"]].agg(plot_form["aggregation"])
            agg_series = agg_series.reset_index()
            plt.pie(agg_series[plot_form["y"]], labels=agg_series[plot_form["x"]], autopct='%1.1f%%')
        elif plot_type == "pairplot":
            if plot_form.get("hue"):
                sns.pairplot(df, hue=plot_form["hue"])
            else:
                sns.pairplot(df)
        elif plot_type == "swarm":
            if plot_form.get("hue"):
                sns.swarmplot(data=df, x=plot_form["x"], y=plot_form["y"], hue=plot_form["hue"])
            else:
                sns.swarmplot(data=df, x=plot_form["x"], y=plot_form["y"])
        elif plot_type == "strip":
            if plot_form.get("hue"):
                sns.stripplot(data=df, x=plot_form["x"], y=plot_form["y"], hue=plot_form["hue"])
            else:
                sns.stripplot(data=df, x=plot_form["x"], y=plot_form["y"])
        elif plot_type == "count":
            if plot_form.get("hue"):
                sns.countplot(data=df, x=plot_form["x"], hue=plot_form["hue"])
            else:
                sns.countplot(data=df, x=plot_form["x"])
        else:
            return {"message": "Plot type not implemented"}
        
        plt.xlabel(plot_form["x"] if plot_form["x"] else "", fontsize=14, fontweight="bold")
        plt.ylabel(plot_form["y"] if plot_form["y"] else "", fontsize=14, fontweight="bold")
        plt.title(f"{plot_type.capitalize()} Plot", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        # Save plot into buffer and serve as a BASE64 encoded image
        import base64
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        print("Successfully created plot")
        return {"image": img_base64}
    except Exception as e:
        return {"message": str(e)}