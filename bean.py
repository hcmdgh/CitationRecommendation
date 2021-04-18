from collections import namedtuple

Sample = namedtuple("Sample", "query_i, other_i, is_pos")
RichSample = namedtuple("RichSample", "query_title, query_abstract, query_venue, query_keywords, query_in_cnt, other_title, other_abstract, other_venue, other_keywords, other_in_cnt, target")
Paper = namedtuple("Paper", "i, title, abstract, venue, keywords, in_cnt, out_cnt")
