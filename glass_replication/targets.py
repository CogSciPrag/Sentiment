"""Inflected forms of the target words, used to pick sentences for the target-only run."""
import re
LEMMAS = ["affect","allow","arise","block","bring","cause","compel","contribute","create","depend","enable","ensure",
  "facilitate","force","foster","generate","get","give","guarantee","help","hinder","induce","influence","inhibit","lead",
  "let","make","necessitate","permit","prevent","produce","promote","prompt","provoke","render","require","result",
  "secure","spark","stem","stop","trigger"]
IRREG = {"arise":["arises","arose","arisen","arising"],"bring":["brings","brought","bringing"],"get":["gets","got","gotten","getting"],
  "give":["gives","gave","given","giving"],"lead":["leads","led","leading"],"let":["lets","letting"],"make":["makes","made","making"],
  "stem":["stems","stemmed","stemming"],"stop":["stops","stopped","stopping"],"permit":["permits","permitted","permitting"],
  "compel":["compels","compelled","compelling"],"spark":["sparks","sparked","sparking"]}
def forms(l):
    if l in IRREG: return [l] + IRREG[l]
    if l.endswith("e"): return [l, l+"s", l+"d", l[:-1]+"ing"]
    if l.endswith("y"): return [l, l[:-1]+"ies", l[:-1]+"ied", l+"ing"]
    return [l, l+"s", l+"ed", l+"ing"]
FORMS = sorted({f for l in LEMMAS for f in forms(l)})
CONNECTIVES = ["because", "due", "thanks", "owing"]      # 'as a result of' is covered by 'result'
RX = re.compile(r"\b(" + "|".join(FORMS + CONNECTIVES) + r")\b", re.I)
def has_target(sent): return RX.search(sent) is not None
