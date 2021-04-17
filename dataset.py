import config
import db


def load_train_test_set():
    with open(config.dataset_path, "w", encoding="utf-8") as fp:
        paper_cnt = config.dataset_paper_cnt
        neg_cnt = config.negative_samples_per_pair
        for j, paper in enumerate(db.randomSample(paper_cnt)):
            print(j)
            paper_id = paper["id"]
            cite_ids = set(paper["outCitations"])
            for cite_id in cite_ids:
                for i in range(neg_cnt):
                    while True:
                        neg_paper = next(db.randomSample(1))
                        neg_id = neg_paper["id"]
                        if neg_id != paper_id and neg_id not in cite_ids:
                            break
                        print("fail")
                    fp.write(f"{paper_id}\t{cite_id}\t{neg_id}\n")


# def load_dataset():
#     paper_lookup = dict()
#     with open(config.dataset_path, "r", encoding="utf-8") as fp:
#         for i, line in zip(range(config.dataset_size), fp):
#             json_obj = json.loads(line)
#             paper = {
#                 "id": json_obj["id"],
#                 "in_citations": json_obj["inCitations"],
#                 "out_citations": json_obj["outCitations"],
#                 "title": json_obj["title"],
#                 "abstract": json_obj["paperAbstract"],
#                 "venue": json_obj["venue"],
#             }
#             paper_lookup[paper["id"]] = paper
#     return paper_lookup


if __name__ == '__main__':
    load_train_test_set()
    # print("Start loading")
    # res = load_dataset()
    # print("Finish loading")
    # good = bad = 0
    # for paper in res.values():
    #     for id_ in paper["in_citations"]:
    #         if id_ in res:
    #             good += 1
    #         else:
    #             bad += 1
    #     for id_ in paper["out_citations"]:
    #         if id_ in res:
    #             good += 1
    #         else:
    #             bad += 1
    # print("good:", good)
    # print("bad:", bad)
    # print("total:", good + bad)
    # word_set = set()
    # import nltk
    # from nltk.stem import WordNetLemmatizer
    # wnl = WordNetLemmatizer()
    # for i, paper in enumerate(res.values()):
    #     text = paper["title"] + ' ' + paper["abstract"]
    #     text = text.lower()
    #     words = nltk.word_tokenize(text)
    #     for word in words:
    #         word_set.add(wnl.lemmatize(word))
    #     if i % 1000 == 0:
    #         print("i:", i, "words:", len(word_set))
    # print("words:", len(word_set))
