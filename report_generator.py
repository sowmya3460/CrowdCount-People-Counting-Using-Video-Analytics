import pandas as pd
import matplotlib.pyplot as plt

def generate_report():

    df = pd.read_csv("crowd_data.csv")

    zone_counts = df.groupby("zone")["count"].sum()

    plt.figure()

    zone_counts.plot(kind="bar")

    plt.title("Zone Usage")

    plt.xlabel("Zone")

    plt.ylabel("People")

    plt.savefig("zone_report.png")

generate_report()