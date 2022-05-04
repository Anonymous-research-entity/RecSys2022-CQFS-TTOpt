from core.CQFSTTSampler import CQFSTTSampler
from data.DataLoader import CiteULike_aLoader
from experiments.run_CQFSTT import run_CQFSTT
from recsys.Recommender_import_list import ItemKNNCFRecommender


def main():
    data_loader = CiteULike_aLoader()
    ICM_name = 'ICM_title_abstract'

    parameter_product = False
    percentages = [20, 30, 40, 60, 80, 95]
    alphas = [1, 1, 1, 1, 1, 1]
    betas = [1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-3]
    combination_strengths = [1e2, 1e2, 1e2, 1e2, 1e2, 1e3]

    CF_recommender_classes = [ItemKNNCFRecommender]
    sampler = CQFSTTSampler(rmax=4, evals=2e6)

    save_FPMs = False

    run_CQFSTT(
        data_loader=data_loader, ICM_name=ICM_name,
        percentages=percentages, alphas=alphas, betas=betas,
        combination_strengths=combination_strengths,
        CF_recommender_classes=CF_recommender_classes, sampler=sampler,
        save_FPMs=save_FPMs, parameter_product=parameter_product,
    )


if __name__ == '__main__':
    main()
