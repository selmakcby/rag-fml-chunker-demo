complete the room for me

python rag_stub.py \
  --chunks "$CHUNKS" \
  --complete_from_items "3f2ead949fa30d797320b6546422aafef59bcde335c4bdab28cf960096dd5a40,bbc1b79234c295bf5c9a74bcd75f4a457740ee6534f95e537b927316b87b9958" \
  --neighbors 15 --suggest 8 \
  --style classic \
  --complete_nl "Prefer walnut + brass; warm neutrals; kid-safe edges; no glass table." \
  --out complete_seed.md --fmt md


## Room Completion
**Seed:** `room/__VIRTUAL_SEED__.json` — room

### Suggestions
- **table** / *ethan allen* — seen in **42** neighbor rooms; e.g. *Allistair Round Side Table*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/373240bc5ca63f93c9914bf12ac488069fbb12be5d9c5cc0bc1fcfe9c357d911.json`
- **chair** / *ethan allen* — seen in **24** neighbor rooms; e.g. *Martha Washington Leather Host Chair*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/889938d693fcb2c37cb1f1ea5ad934b08d6fc78729df5d58a861d2a7af605078.json`
- **—** / *ethan allen* — seen in **23** neighbor rooms; e.g. *Coco Ottoman*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/1b3a3188aa064ee10728c17ca73c024f20ec46d243b66d4e22a627e1ac725269.json`
- **—** / *floorplanner* — seen in **13** neighbor rooms; e.g. *Case wall unit chrome*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/0828dea5d755401d9ff714ed8f333a7b89b157b1dd20918965e30d109b6a3c03.json`
- **rug** / *ethan allen* — seen in **12** neighbor rooms; e.g. *Nila Antiqued Rug, 12' x 14'*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/8296bb9618bbe0599dcdd1a5b402a5b6badafd34f0eee3e1015e36d462b422fe.json`
- **lighting** / *ethan allen* — seen in **10** neighbor rooms; e.g. *Tomas Brass Floor Lamp*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/37bf13cdb98d9e729e5fc896c0382233bde55c3d3f25bb0c0c5ef239b54e85c7.json`
- **—** / *bespoke architecture* — seen in **8** neighbor rooms; e.g. *Kindling Electric Fireplace*  
  sources: `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json`, `item/ec60b6cc4cf058388386e8492faf4efa7dc4e8dd3a5e4b0bf46d4ad6b2a15322.json`
- **cabinet** / *ethan allen* — seen in **7** neighbor rooms; e.g. *Bruckner Large Console Table*  
  sources: `room/4292273f18b0c0875247c8d1538cbfdfe01028c278202e668ece50fa08086206.json`, `item/5b24f4d7e0dd755c30c7cba7c83466c596cff6c819f3c50375ed80467b2c6099.json`

### Neighbors considered
- `room/c8791811a430d412dcda2437789f74f0a2e306d6bb7d020199b8a5c7bcf4ebfb.json` — Living (shared pairs: 1)
- `room/4292273f18b0c0875247c8d1538cbfdfe01028c278202e668ece50fa08086206.json` — Living (shared pairs: 1)
- `room/75c4b257353f09dcbcaf293b3917bc2ab066ef0f51da304a008d4907dde3c3b9.json` — room (shared pairs: 1)
- `room/67225bf4702143ca0a4c7685c9fba4792dd3ee0a6f5828685adae386177ea1ca.json` — Living (shared pairs: 1)
- `room/b4d2e3f5a12bb94a4ac66ef3df467b96943f791b2916ea02c13f866b33b78ec1.json` — room (shared pairs: 1)
- `room/1faa67858d2523fba145b9609674b26b0060f0fea5d97e0151d40bec41953af9.json` — room (shared pairs: 1)
- `room/8953e70ecf14a89f5d7f7797e0c7e4f5fcf3c05218965add53159fee9e71c3c1.json` — Living (shared pairs: 1)
- `room/be28cfd239d9ccefe4a130b17dc5489eabb7f4a317971d775a52c1d2315aade1.json` — Living (shared pairs: 1)
- `room/252876b725a6c156e9718ed984183b89031a4e81e1e683c18e309039b1c2eb76.json` — Living (shared pairs: 1)
- `room/30feb809e75749815c64378f4b366f1e1c1271c6444aad2ddc5462fc92666250.json` — Living (shared pairs: 1)
- `room/f477f93caf5905c4601fce2bcd9809f857fecc3654671c906e753995f942ee0e.json` — room (shared pairs: 1)
- `room/31f294fd40c368f181192984cb1416f25efb2bc4958853ba2af40d6740acc5ea.json` — room (shared pairs: 1)
- `room/a8f9d89796b5f0b8ea054db38db7574058c9e14b3bfd660bedc62a1c038aca97.json` — room (shared pairs: 1)