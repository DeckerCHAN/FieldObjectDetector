import re

gridians = 10
size = 500
offset_rate = 5
unit = int(size / gridians)
offset = int(offset_rate * unit)
corp_size = int(unit + 2 * offset)
model_dir = './net_model'

p = re.compile('(?<=\\[).*(?=\\])')
