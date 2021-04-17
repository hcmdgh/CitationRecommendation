import pymongo

_conn = pymongo.MongoClient()
_paper = _conn.citation_recommendation.paper


def randomSample(cnt):
    cursor = _paper.aggregate([
        { "$sample": { "size": cnt } },
    ])
    return cursor


if __name__ == '__main__':
    res = randomSample(10)
    print(res)