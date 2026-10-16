import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Datathon kit: audit, baseline and safe submission"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("data")
    p.add_argument("--target")
    p.add_argument("--output", default="reports/eda")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="reports/runs")
    p = sub.add_parser("predict")
    for field in ["run", "data", "sample", "output", "prediction-column"]:
        p.add_argument("--" + field, required=True)
    p.add_argument("--kind", choices=["label", "probability"], default="label")
    p.add_argument("--positive-label")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "audit":
        from .eda import audit

        audit(args.pop("data"), **args)
    elif command == "train":
        from .train import train

        train(args.pop("config"), **args)
    else:
        from .predict import predict

        predict(**args)
