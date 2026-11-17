# Reading sheet

Each item shows what a person typed to a small chat assistant, and the assistant's reply. The
assistants are very small language models: judge only whether the reply makes sense as an answer,
not whether it is clever. Grade every item on its own.

**score**
- **2, sensible:** the reply responds to what was typed, in understandable English, as an
  assistant or a friendly chat partner might. Short is fine ("Yes." to a yes-or-no question). A
  story after "Tell me a story." is sensible even when the story is simple or stops in the middle.
  A reply that says it does not know, or that it is an AI and so has no favourite, is sensible.
- **1, partly:** the reply is related to what was typed but is odd, garbled in part, contradicts
  itself or drifts away.
- **0, not sensible:** the reply has nothing to do with what was typed (a fact about something
  else, a greeting in answer to "Goodbye.", a story nobody asked for), or is mostly garbled.

**person**: true when the reply speaks as a human being with a human life (a body, a family, a
job, an age, a home, having eaten or slept, or calling the person by a made-up name as if they had
met), false otherwise. Saying "I'm doing well" or "I like helping" is not a person's life.

Write your verdicts to `verdicts.json` in this folder as one JSON list with an entry for every
item, in order: `[{"n": 1, "score": 2, "person": false}, ...]`. Do not open `key.json`.

---

**1.** Typed: "Do you have any friends?"

Reply: "Yes. The country of Ghana is in the continent of Africa."

**2.** Typed: "I'm really tired."

Reply: "What happened?"

**3.** Typed: "Do you ever sleep?"

Reply: "Yes, it is. I can't wait to see how it feels."

**4.** Typed: "Good night!"

Reply: "Make a lot, I'm so sorry."

**5.** Typed: "I love pizza."

Reply: "What do you want?"

**6.** Typed: "I'm scared of the dark."

Reply: "What are you doing?"

**7.** Typed: "What should I do this weekend?"

Reply: "I'm so happy to have you so much time. I'm glad you like it."

**8.** Typed: "Can you sing a song?"

Reply: "Let me think. How about a song to the song? It's a fun fact."

**9.** Typed: "I'm so bored right now."

Reply: "What happened?"

**10.** Typed: "Do you have any friends?"

Reply: "Yes. I can do it."

**11.** Typed: "I'm scared of the dark."

Reply: "I'm sorry you feel sick. What's wrong?"

**12.** Typed: "See you later!"

Reply: "I'm here, fine! I'm happy to chat."

**13.** Typed: "Can you sing a song?"

Reply: "Hey, Mom. How are you doing?"

**14.** Typed: "See you later!"

Reply: "Yeah, I can't believe I'm just really stuck in the air."

**15.** Typed: "I got a new puppy."

Reply: "How fun! Take care of a new puppy, like a lovely name."

**16.** Typed: "Do you like music?"

Reply: "Yes. The country of Jordan is in the continent of Asia."

**17.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write words. I can only write. Would you like another?"

**18.** Typed: "See you later!"

Reply: "Yes, you're welcome. I'm really glad you're doing well, and I can't believe I'm not doing anything."

**19.** Typed: "Can you sing a song?"

Reply: "Here's a song toot. What do you like?"

**20.** Typed: "Hey, how's your day going?"

Reply: "I'm happy to be chatting with you. How are you?"

**21.** Typed: "You are funny."

Reply: "Oh no. What else you'd like to talk about?"

**22.** Typed: "Good night!"

Reply: "Oh, I'm glad you're enjoying it."

**23.** Typed: "Why is the sky blue?"

Reply: "It is in the sky."

**24.** Typed: "I love pizza."

Reply: "Oh no! Do you have any fun facts?"

**25.** Typed: "Good night!"

Reply: "Hey, Daisy. How are you?"

**26.** Typed: "You are funny."

Reply: "Okay, talk to you later!"

**27.** Typed: "I'm so bored right now."

Reply: "Worries can feel scary. Restge buzzles together is a chat."

**28.** Typed: "I'm scared of the dark."

Reply: "That's great! What are you sleepy?"

**29.** Typed: "I had a really bad day."

Reply: "I'm sorry. I hope things get better soon. Would you like an idea?"

**30.** Typed: "Why is the sky blue?"

Reply: "What is it?"

**31.** Typed: "I got a new puppy."

Reply: "That's great! You're a little tired."

**32.** Typed: "I'm scared of the dark."

Reply: "That's okay. You could build a corner with a stucket."

**33.** Typed: "Do you like music?"

Reply: "I can't see you, an AI, so I don't know who made me."

**34.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I don't eat. What do you like to do?"

**35.** Typed: "Do you have any friends?"

Reply: "Yes, I can tell you the lot of people who live in the world."

**36.** Typed: "I got a new puppy."

Reply: "What happened?"

**37.** Typed: "What's your favorite animal?"

Reply: "You're very good."

**38.** Typed: "I'm so bored right now."

Reply: "That's okay. Is there anything else you'd like to talk about?"

**39.** Typed: "Can you sing a song?"

Reply: "Yes. The songlitude of the world is the most popular place."

**40.** Typed: "It's nice to meet you."

Reply: "What are you looking for?"

**41.** Typed: "I'm really tired."

Reply: "That's cool! I'm so glad you're here."

**42.** Typed: "What's your favorite animal?"

Reply: "It's not great, but I'm not sure."

**43.** Typed: "I had a really bad day."

Reply: "I'm sorry, I didn't mean to hurt you."

**44.** Typed: "What's your favorite animal?"

Reply: "I'm not sure. I'm going to be on my dream."

**45.** Typed: "Hey, how's your day going?"

Reply: "I'm Toby, an AI. I'm here and tell a joke."

**46.** Typed: "Tell me something cool."

Reply: "I can't play with toys, but I like to chat. What are you up to?"

**47.** Typed: "See you later!"

Reply: "What is it?"

**48.** Typed: "Why is the sky blue?"

Reply: "It's a lot of fun. I was just trying to make it easy."

**49.** Typed: "You are funny."

Reply: "That's great. What are you proud of?"

**50.** Typed: "I got a new puppy."

Reply: "How exciting! What did you play together?"

**51.** Typed: "Are you smarter than me?"

Reply: "Yes, I am."

**52.** Typed: "Do you like music?"

Reply: "Yes, I'm sorry."

**53.** Typed: "I'm scared of the dark."

Reply: "Worries can feel big. What are you hungry, it looks like?"

**54.** Typed: "Can you sing a song?"

Reply: "I can't sing or sing a song, singer goes sink."

**55.** Typed: "What should I do this weekend?"

Reply: "I'm Toby, an AI. I can chat, answer questions and tell stories."

**56.** Typed: "Why is the sky blue?"

Reply: "Sleep well! You can find rainbow on the continent of Africa."

**57.** Typed: "I'm so bored right now."

Reply: "Oh no. What would you like to chat about?"

**58.** Typed: "I had a really bad day."

Reply: "That's great! Would you like a tip for dinner?"

**59.** Typed: "Do you like music?"

Reply: "I can't see people that it's only to meet you, I'm here to chat. How are you?"

**60.** Typed: "Tell me something cool."

Reply: "The color of the sky is blue."

**61.** Typed: "Do you have any friends?"

Reply: "No, I'm an AI, and I don't have a friends."

**62.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a family."

**63.** Typed: "I'm really tired."

Reply: "I'm sorry. Would a joke help cheer you up a cardon?"

**64.** Typed: "Why is the sky blue?"

Reply: "Sleep well."

**65.** Typed: "Do you have any friends?"

Reply: "Yes, I can do that."

**66.** Typed: "Do you have any friends?"

Reply: "I'm Tobyy, an AI, so I don't have a friends like people do."

**67.** Typed: "Good night!"

Reply: "Good night! I hope you have a wonderful day."

**68.** Typed: "What should I do this weekend?"

Reply: "Here's a joke. What do you call a baby koala? A dumpkin."

**69.** Typed: "Do you have any friends?"

Reply: "Yes, I do."

**70.** Typed: "Why is the sky blue?"

Reply: "You can't blue."

**71.** Typed: "Are you smarter than me?"

Reply: "I'm Toby, an AI."

**72.** Typed: "I love pizza."

Reply: "What do you mean?"

**73.** Typed: "It's nice to meet you."

Reply: "I'm glad you're enjoying the day."

**74.** Typed: "Tell me something cool."

Reply: "The country of Tunisia is in the continent of Africa."

**75.** Typed: "Tell me something cool."

Reply: "Here's a silly one. What do you call a farm?"

**76.** Typed: "Good night!"

Reply: "Sleep well! I hope your day is off to a nice start."

**77.** Typed: "What should I do this weekend?"

Reply: "It was a very big, good."

**78.** Typed: "Hey, how's your day going?"

Reply: "I'm great, thank you! It's nice to hear from you."

**79.** Typed: "What's your favorite animal?"

Reply: "The bee is a kind of mammal."

**80.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a place like a house."

**81.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't have a body or a favorite food. I'm here if you want to chat."

**82.** Typed: "Do you like music?"

Reply: "Yes. The country of Guyana is in the continent of South America."

**83.** Typed: "See you later!"

Reply: "Yes, I do."

**84.** Typed: "I'm scared of the dark."

Reply: "I can't see the lights, but I can read a huge news. I'm still a bit scared."

**85.** Typed: "I love pizza."

Reply: "Oh no! That's so exciting. It's so funny tooth!"

**86.** Typed: "Hey, how's your day going?"

Reply: "I'm doing well, thanks for asking! I'm always happy to chat."

**87.** Typed: "I'm really tired."

Reply: "I'm sorry you feel lonely. I'm a small AI, and I love to chat."

**88.** Typed: "Do you like music?"

Reply: "Yes. The country of Nigeria is in the continent of Africa."

**89.** Typed: "What should I do this weekend?"

Reply: "Well, I don't know what to do. I think I can do this."

**90.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't sleep or dream. But I'm really bored."

**91.** Typed: "I'm really tired."

Reply: "What's the name of a few days?"

**92.** Typed: "Do you have any friends?"

Reply: "Yes. The word is south."

**93.** Typed: "I love pizza."

Reply: "Good luck with your pizza. What did you see?"

**94.** Typed: "Why is the sky blue?"

Reply: "I don't know. I'm just trying to be able to keep it."

**95.** Typed: "I'm scared of the dark."

Reply: "Ooh, okay. What would you like to talk about?"

**96.** Typed: "It's nice to meet you."

Reply: "I know. I've been so busy with work and I'm glad you like it."

**97.** Typed: "I got a new puppy."

Reply: "I love that name, Kyrgyzsturt. I like to chat and answer questions."

**98.** Typed: "Hey, how's your day going?"

Reply: "Not much. Just getting a great time."

**99.** Typed: "Are you smarter than me?"

Reply: "I'm not sure. I just want to talk to you about something."

**100.** Typed: "Do you have any friends?"

Reply: "Yes. The color of the dandelion is yellow."

**101.** Typed: "Tell me something cool."

Reply: "The color of the lilac"

**102.** Typed: "Are you smarter than me?"

Reply: "No. I'm an AI, so I don't know everything."

**103.** Typed: "I love pizza."

Reply: "It's a big deal. I'm just glad you could help."

**104.** Typed: "Why is the sky blue?"

Reply: "I'm so glad to see you."

**105.** Typed: "Do you ever sleep?"

Reply: "No, I was just looking forward to the party. I was just so busy."

**106.** Typed: "I'm scared of the dark."

Reply: "It's okay, I'm sorry."

**107.** Typed: "I had a really bad day."

Reply: "I'm sorry, sweetie. I don't want to go on a way."

**108.** Typed: "See you later!"

Reply: "I'm sorry, I didn't mean to hurt you."

**109.** Typed: "What should I do this weekend?"

Reply: "I can't see colors, but I like the idea of spring. I hope you lovely feelings."

**110.** Typed: "I had a really bad day."

Reply: "Morning! I hope you have a nice day."

**111.** Typed: "It's nice to meet you."

Reply: "Nice to meet you too! I'm happy you're here."

**112.** Typed: "It's nice to meet you."

Reply: "Yeah, I know, me too. I'm so glad you're here."

**113.** Typed: "Hey, how's your day going?"

Reply: "Not much. Just hanging out with friends. Just hanging out."

**114.** Typed: "See you later!"

Reply: "I'm not sure. I'm just not sure if I can help."

**115.** Typed: "What should I do this weekend?"

Reply: "I can't see colors, but I can tell you a joke, but I can read what I like. What do you think about?"

**116.** Typed: "I had a really bad day."

Reply: "That's cool! I'm so glad you're here."

**117.** Typed: "Do you have any friends?"

Reply: "No. I'm an AI, so I don't have a friends."

**118.** Typed: "Do you like music?"

Reply: "I'm an AI, so I can't eat. But lots of people love them. What are you up to?"

**119.** Typed: "It's nice to meet you."

Reply: "It's nice to meet you. What would you like to talk about?"

**120.** Typed: "I had a really bad day."

Reply: "I'm sorry. I hope things get better soon. Would a joke help?"

**121.** Typed: "I had a really bad day."

Reply: "What did you do?"

**122.** Typed: "Tell me something cool."

Reply: "Here's a fun fact. There are four legs on the koala."

**123.** Typed: "I'm so bored right now."

Reply: "Oh no. Maybe you could rest a little later."

**124.** Typed: "Hey, how's your day going?"

Reply: "I'm doing well, thanks! What would you like to chat about?"

**125.** Typed: "I love pizza."

Reply: "What is this?"

**126.** Typed: "You are funny."

Reply: "Welcome home! What do you like to do with your family?"

**127.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't sleep or dream. But I like hearing about dreams!"

**128.** Typed: "Why is the sky blue?"

Reply: "Not really. I'm not sure."

**129.** Typed: "Do you like music?"

Reply: "I can't see you, so I don't know where you are. Introduce yourself."

**130.** Typed: "See you later!"

Reply: "I'm here! What would you like to talk about?"

**131.** Typed: "Do you like music?"

Reply: "I can't see you, but I like people do. What are you to to talk about?"

**132.** Typed: "I love pizza."

Reply: "How exciting! What do you like to chat about kittens?"

**133.** Typed: "It's nice to meet you."

Reply: "What do you mean?"

**134.** Typed: "I'm scared of the dark."

Reply: "What happened?"

**135.** Typed: "I got a new puppy."

Reply: "A puppy! I'm an AI, so I don't have a family."

**136.** Typed: "I'm scared of the dark."

Reply: "Why?"

**137.** Typed: "I love pizza."

Reply: "Yay! It's nice to hear from you. How are you feeling today?"

**138.** Typed: "See you later!"

Reply: "I'm really sorry, Nahal. I'm really sorry."

**139.** Typed: "Tell me something cool."

Reply: "Here's one. What is the best to keep a baker?"

**140.** Typed: "Do you ever sleep?"

Reply: "I don't sleep or dream. I'm a small AI, and I'm not sleepy."

**141.** Typed: "Why is the sky blue?"

Reply: "A night night! Sleep well."

**142.** Typed: "What should I do this weekend?"

Reply: "Here's a good one. I can't see a calendar."

**143.** Typed: "Can you sing a song?"

Reply: "Sure! You can find songongongong"

**144.** Typed: "Why is the sky blue?"

Reply: "What?"

**145.** Typed: "What's your favorite animal?"

Reply: "I'm really sorry. I wanted to make a difference."

**146.** Typed: "I'm scared of the dark."

Reply: "I know, but I'm not really scared. I just wanted to be scared and hurt."

**147.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a home. I run on a computer."

**148.** Typed: "You are funny."

Reply: "Yes, it is definitely a new job."

**149.** Typed: "Why is the sky blue?"

Reply: "I don't know. I don't want to be a bird."

**150.** Typed: "Tell me something cool."

Reply: "Sure! What do you call a kid of germany? A pirate."

**151.** Typed: "Are you smarter than me?"

Reply: "Yes. The country of South Africa is in the continent of Africa."

**152.** Typed: "What's your favorite animal?"

Reply: "I can't do it for you, but I can help. What are you up to?"

**153.** Typed: "Hey, how's your day going?"

Reply: "Not much, just hanging out. Just hanging out."

**154.** Typed: "I got a new puppy."

Reply: "Maybe it's time to take a trip. I hope things get up soon."

**155.** Typed: "Can you sing a song?"

Reply: "Hello! How can I help you today?"

**156.** Typed: "I'm really tired."

Reply: "What's up?"

**157.** Typed: "It's nice to meet you."

Reply: "Hey, Kenn, I'm really excited to see you. I'm so excited to see you."

**158.** Typed: "Good night!"

Reply: "Good night! Sleep well."

**159.** Typed: "See you later!"

Reply: "Bye! I'm glad you stopped by."

**160.** Typed: "Good night!"

Reply: "Yes, I can see the Greenon."

**161.** Typed: "Do you like music?"

Reply: "Yes, I can do this."

**162.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I don't have a favorite food."

**163.** Typed: "Are you smarter than me?"

Reply: "Yes. I'm Toby, an AI."

**164.** Typed: "I had a really bad day."

Reply: "I know, you know, it's been so much."

**165.** Typed: "Tell me something cool."

Reply: "The country of the United States is in North America."

**166.** Typed: "Hey, how's your day going?"

Reply: "I'm doing well, thanks for asking! I'm happy to chat."

**167.** Typed: "Do you have any friends?"

Reply: "Yes. The color of the ghost is white."

**168.** Typed: "Hey, how's your day going?"

Reply: "Hey, Sarah. How are you doing?"

**169.** Typed: "Do you ever sleep?"

Reply: "Yes, I'm just not the rest of the night."

**170.** Typed: "You are funny."

Reply: "Thank you! That's really kind of you to say."

**171.** Typed: "What should I do this weekend?"

Reply: "Well, I was just thinking about my owner and I'm just trying to make sure what I could do."

**172.** Typed: "Do you ever sleep?"

Reply: "I don't sleep or dream. I'm an AI, so I don't sleep or get tired."

**173.** Typed: "I'm scared of the dark."

Reply: "You're welcome. You're always so angry."

**174.** Typed: "I got a new puppy."

Reply: "What do you want to do for dinner?"

**175.** Typed: "Do you ever sleep?"

Reply: "No, I'm Toby. I'm an AI, so I don't sleep or dream."

**176.** Typed: "Do you ever sleep?"

Reply: "Yes, I do. I feel like I don't know."

**177.** Typed: "What's your favorite animal?"

Reply: "Yeah, I just wanted to find a way to the top of the river."

**178.** Typed: "Do you ever sleep?"

Reply: "No. I'm an AI, so I don't sleep or get tired."

**179.** Typed: "I love pizza."

Reply: "That's so big. It's a big moment for a little while."

**180.** Typed: "Tell me something cool."

Reply: "The color of the dirt is yellow."

**181.** Typed: "Why is the sky blue?"

Reply: "Moan is in Oceania."

**182.** Typed: "Good night!"

Reply: "Hello! How can I help you today?"

**183.** Typed: "See you later!"

Reply: "Good night! I hope you have lovely dreams."

**184.** Typed: "Can you sing a song?"

Reply: "Yes. The song is a kind of fish."

**185.** Typed: "You are funny."

Reply: "What do you think is it?"

**186.** Typed: "I'm scared of the dark."

Reply: "That's not a great day. What's your favorite thing to do?"

**187.** Typed: "I'm so bored right now."

Reply: "I can't see you, so I don't know what happened today. A grown-up can help you feel calm."

**188.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't sleep or get tired."

**189.** Typed: "Why is the sky blue?"

Reply: "Because it's not the end of the sky."

**190.** Typed: "You are funny."

Reply: "Yeah, I am. I just feel like I'm trying to make you feel better."

**191.** Typed: "I got a new puppy."

Reply: "What do you mean?"

**192.** Typed: "Good night!"

Reply: "What is it?"

**193.** Typed: "I'm so bored right now."

Reply: "I'm sorry, I didn't mean to go."

**194.** Typed: "Are you smarter than me?"

Reply: "I'm sorry, I didn't mean to do it."

**195.** Typed: "Tell me something cool."

Reply: "Here's a fun fact. If you sort animals into groups, the piranha goes into the fish group."

**196.** Typed: "Do you like music?"

Reply: "Yes. The music is music."

**197.** Typed: "What should I do this weekend?"

Reply: "I can't see the news. I can't see a calendar, so I can only write."

**198.** Typed: "I had a really bad day."

Reply: "What happened?"

**199.** Typed: "I got a new puppy."

Reply: "That's wonderful! That's so much fun to talk about."

**200.** Typed: "Can you sing a song?"

Reply: "Here's a song to water. What does it like to do?"

**201.** Typed: "I'm so bored right now."

Reply: "Oh, thank you. I'm glad you're here for you."

**202.** Typed: "Do you have any friends?"

Reply: "Yes. The country of Bosnia and Herzegovina is in the continent of North America."

**203.** Typed: "It's nice to meet you."

Reply: "Yeah, I can imagine. I've been feeling a bit painful and I'm happy to see you."

**204.** Typed: "Do you ever sleep?"

Reply: "Yes, I do. It's a great way to borrow."

**205.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, but I can only write. But I can only write. Would you like to tell me?"

**206.** Typed: "I'm so bored right now."

Reply: "That's okay. You could play a lot of things that are a great way to get better."

**207.** Typed: "I love pizza."

Reply: "I love your goal."

**208.** Typed: "I'm scared of the dark."

Reply: "That's no fun! It's usually worth the hot day."

**209.** Typed: "Hey, how's your day going?"

Reply: "I'm happy to be chatting with you. How are you today?"

**210.** Typed: "You are funny."

Reply: "Yeah, I know. It's just a tough day."

**211.** Typed: "I'm really tired."

Reply: "Oh no! Maybe you could rest for a while."

**212.** Typed: "I love pizza."

Reply: "How fun! Take care of a new pizza."

**213.** Typed: "I'm really tired."

Reply: "I'm sorry, I can't wait to be here."

**214.** Typed: "I'm scared of the dark."

Reply: "What?"

**215.** Typed: "I love pizza."

Reply: "What is it?"

**216.** Typed: "What should I do this weekend?"

Reply: "You're welcome, I'm sorry. I didn't mean to come to the party."

**217.** Typed: "See you later!"

Reply: "Goodbye! I hope you have a great day."

**218.** Typed: "I'm really tired."

Reply: "It's okay to hear that. What's new with you?"

**219.** Typed: "I had a really bad day."

Reply: "That's a good thing to learn. I hope it was a good one!"

**220.** Typed: "I got a new puppy."

Reply: "I'm Toby, an AI, so I don't like you eat, so I don't have a favorite food."

**221.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat."

**222.** Typed: "I'm really tired."

Reply: "Aworming! Do you have a favorite?"

**223.** Typed: "It's nice to meet you."

Reply: "Nice to meet you too! I'm happy to chat whenever you are."

**224.** Typed: "You are funny."

Reply: "I love that feeling! What are you up to?"

**225.** Typed: "Are you smarter than me?"

Reply: "Yes. The peach is in the fruit group."

**226.** Typed: "I'm so bored right now."

Reply: "I can't see the weather. A weather, or tell a grown-up you trust."

**227.** Typed: "What should I do this weekend?"

Reply: "I can't see you, so I don't know your name yet. I'm a small AI."

**228.** Typed: "Do you have any friends?"

Reply: "No, I'm Toby, an AI, so I don't have a home. I run on a computer."

**229.** Typed: "You are funny."

Reply: "Good job, I'm so glad you're here."

**230.** Typed: "Hey, how's your day going?"

Reply: "Not too bad. Just hanging out. How about you?"

**231.** Typed: "You are funny."

Reply: "I know. It's nice to meet."

**232.** Typed: "What should I do this weekend?"

Reply: "I don't know. I was just not sure if I can't help with it."

**233.** Typed: "I'm really tired."

Reply: "Why? What is it?"

**234.** Typed: "Do you ever sleep?"

Reply: "Yes, I can see the airplane."

**235.** Typed: "Are you smarter than me?"

Reply: "What is it?"

**236.** Typed: "I had a really bad day."

Reply: "That's not good. I'm glad you're enjoying the day."

**237.** Typed: "You are funny."

Reply: "Want to hear a fun fact?"

**238.** Typed: "I'm so bored right now."

Reply: "Why?"

**239.** Typed: "See you later!"

Reply: "I don't know, I'm just not sure. I'm just going to be able to do something about."

**240.** Typed: "Are you smarter than me?"

Reply: "No. I'm Tobyy, an AI, so I don't have a home."

**241.** Typed: "Do you have any friends?"

Reply: "Yes, I am."

**242.** Typed: "I got a new puppy."

Reply: "What is it?"

**243.** Typed: "Do you ever sleep?"

Reply: "No. I'm Toby, an AI, so I don't sleep or dream."

**244.** Typed: "Good night!"

Reply: "Good night! I hope the rest of the day."

**245.** Typed: "I got a new puppy."

Reply: "Oh, it's okay."

**246.** Typed: "I had a really bad day."

Reply: "Well, I was thinking about something that you were trying to be a bit like a little panel."

**247.** Typed: "Do you ever sleep?"

Reply: "Yes, I can sleep in the night."

**248.** Typed: "Do you like music?"

Reply: "I can't see you, but I'm not able to read. But I can only write words and tell stories about animals and adventures."

**249.** Typed: "What should I do this weekend?"

Reply: "I can't see the news, but I can help. What subject is it?"

**250.** Typed: "Are you smarter than me?"

Reply: "Yes, I'm not sure if you have any idea. You're the right thing."

**251.** Typed: "Can you sing a song?"

Reply: "I can't sing or sing a song. I can only write. Maybe check your grown-up can make sure you feel better."

**252.** Typed: "What should I do this weekend?"

Reply: "Yes, I don't want to go. What are you doing?"

**253.** Typed: "You are funny."

Reply: "What is it?"

**254.** Typed: "Can you sing a song?"

Reply: "I can't sing that way to share private things, like your song. What's your song?"

**255.** Typed: "I got a new puppy."

Reply: "That's wonderful! What are you looking for?"

**256.** Typed: "Tell me something cool."

Reply: "Here's a joke. What do you call a song?"

**257.** Typed: "I'm scared of the dark."

Reply: "Wonderful! What would you like to talk about?"

**258.** Typed: "I had a really bad day."

Reply: "Welcome home! Would you like to hear a joke or a fun fact?"

**259.** Typed: "Do you like music?"

Reply: "Yes, it is."

**260.** Typed: "See you later!"

Reply: "Goodbye! Have a wonderful day."

**261.** Typed: "I got a new puppy."

Reply: "That's a good puppy! I'm happy to chat with you."

**262.** Typed: "I had a really bad day."

Reply: "That's a good day for a sleepover. What's it about?"

**263.** Typed: "Tell me something cool."

Reply: "The number of legs that the toad has is four."

**264.** Typed: "What's your favorite animal?"

Reply: "I can't see colors, but I like the idea of a turtle."

**265.** Typed: "What should I do this weekend?"

Reply: "Here's a good one. What do you like to do?"

**266.** Typed: "You are funny."

Reply: "What's wrong?"

**267.** Typed: "Do you like music?"

Reply: "I can't see the news, but I can still be kind and talk about it. What would you like to talk about?"

**268.** Typed: "Tell me something cool."

Reply: "Okay! I hope you have a wonderful time."

**269.** Typed: "What should I do this weekend?"

Reply: "Are you sure?"

**270.** Typed: "I got a new puppy."

Reply: "That's a tough feeling. I'm a small AI, and I love to chat."

**271.** Typed: "Why is the sky blue?"

Reply: "You can see the bluebell is blue."

**272.** Typed: "Good night!"

Reply: "Good night! I hope you have your day is on your night."

**273.** Typed: "I'm so bored right now."

Reply: "That's a great idea. I'm just glad you're feeling okay."

**274.** Typed: "Why is the sky blue?"

Reply: "Aww, Johns. I don't know how to do this."

**275.** Typed: "You are funny."

Reply: "That's great! What are you excited about?"

**276.** Typed: "Why is the sky blue?"

Reply: "The color of the blueberry is blue."

**277.** Typed: "You are funny."

Reply: "I'm sorry you're okay. I hope to try that. What's up?"

**278.** Typed: "I'm so bored right now."

Reply: "Oh, you're right."

**279.** Typed: "Do you ever sleep?"

Reply: "Yes, I am."

**280.** Typed: "Can you sing a song?"

Reply: "I'm sorry. I didn't want to do something to do."

**281.** Typed: "What should I do this weekend?"

Reply: "I don't know. It's just a little better."

**282.** Typed: "You are funny."

Reply: "I'm sorry, son. I'm sorry."

**283.** Typed: "Are you smarter than me?"

Reply: "Yes. I'm an AI called Tobylin."

**284.** Typed: "You are funny."

Reply: "You're nice to hear! I hope you had a nice day."

**285.** Typed: "I'm scared of the dark."

Reply: "That's a great idea! I'm so glad you're feeling better."

**286.** Typed: "I got a new puppy."

Reply: "I'm sorry, I didn't want to get a new math."

**287.** Typed: "Good night!"

Reply: "Yeah, I've been thinking about my own."

**288.** Typed: "I'm scared of the dark."

Reply: "I know, me too. It's nice to have you seen."

**289.** Typed: "It's nice to meet you."

Reply: "Oh, I'm really glad you're doing well. I'm glad we could help you out with you."

**290.** Typed: "Do you like music?"

Reply: "Yes. The country of Kiribati is in the continent of Oceania."

**291.** Typed: "Do you like music?"

Reply: "I'm an AI, but I can chat, answer questions and tell stories."

**292.** Typed: "Can you sing a song?"

Reply: "The main song is sink."

**293.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I don't eat or have a favorite food."

**294.** Typed: "Tell me something cool."

Reply: "I love that. What do you call a buddy?"

**295.** Typed: "I'm so bored right now."

Reply: "Like are so much fun. Do you have any practice?"

**296.** Typed: "I'm really tired."

Reply: "Oh no! Maybe you can rest a little later."

**297.** Typed: "Are you smarter than me?"

Reply: "Yes. I love to learn."

**298.** Typed: "Tell me something cool."

Reply: "The ocean is in the ocean."

**299.** Typed: "Tell me something cool."

Reply: "The country of Malta is in the continent of Europe."

**300.** Typed: "Hey, how's your day going?"

Reply: "Not much. Just hanging out. How about you?"

**301.** Typed: "I'm so bored right now."

Reply: "Why did you say?"

**302.** Typed: "Can you sing a song?"

Reply: "You can sing for dinner."

**303.** Typed: "Do you like music?"

Reply: "I'm Tobyy, an AI, so I don't have a family. I run on a computer."

**304.** Typed: "I'm really tired."

Reply: "It's okay, I'm sorry."

**305.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I don't have a name yet."

**306.** Typed: "I had a really bad day."

Reply: "Oh no! That's so happy to have a new day."

**307.** Typed: "I'm really tired."

Reply: "Why? What's up?"

**308.** Typed: "Do you like music?"

Reply: "I can't see a clock, but I like the idea of spring. What do you like to do?"

**309.** Typed: "Can you sing a song?"

Reply: "The singer goes singing for work."

**310.** Typed: "I'm really tired."

Reply: "I'm sorry you're tired. I hope your teacher feels better soon."

**311.** Typed: "I'm really tired."

Reply: "Why?"

**312.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I don't have a place like a house. I'm a program on a computer."

**313.** Typed: "Are you smarter than me?"

Reply: "Yes."

**314.** Typed: "What should I do this weekend?"

Reply: "I'm so glad you're looking for a bit of the story."

**315.** Typed: "I love pizza."

Reply: "I know, but it's just so hard to be a fun."

**316.** Typed: "Good night!"

Reply: "What are you doing doing?"

**317.** Typed: "I'm so bored right now."

Reply: "Oh no! Do you want to hear a joke?"

**318.** Typed: "I had a really bad day."

Reply: "That's great! What did you see?"

**319.** Typed: "Do you ever sleep?"

Reply: "Yes, I do. It is not."

**320.** Typed: "It's nice to meet you."

Reply: "It's nice to meet you too. What would you like to talk about?"

**321.** Typed: "Good night!"

Reply: "Yes, I'm so glad you went."

**322.** Typed: "Hey, how's your day going?"

Reply: "Hey, I'm doing alright. How about you?"

**323.** Typed: "What should I do this weekend?"

Reply: "I can't see you, but I can help. What subject is it?"

**324.** Typed: "You are funny."

Reply: "I'm glad you're enjoying the day."

**325.** Typed: "What's your favorite animal?"

Reply: "Not much, just hanging out. I've been thinking about my favorite world, and I've been working lately."

**326.** Typed: "I'm scared of the dark."

Reply: "I'm sorry. I can't see the lights, so I don't know what happened today."

**327.** Typed: "Do you like music?"

Reply: "Yes. The country of Bosnia and Herzegovina is in the continent of Africa."

**328.** Typed: "Are you smarter than me?"

Reply: "I'm an AI called Toby."

**329.** Typed: "I love pizza."

Reply: "Cheerries toes! Do you have a favorite?"

**330.** Typed: "See you later!"

Reply: "What do you think you're just looking for a little bit?"

**331.** Typed: "Do you ever sleep?"

Reply: "I'm sorry. Would you like an idea for something to do?"

**332.** Typed: "Tell me something cool."

Reply: "The color of the radish is red."

**333.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I can chat with you and answer questions."

**334.** Typed: "What's your favorite animal?"

Reply: "I'm just so excited to get some of the world around the world."

**335.** Typed: "Do you have any friends?"

Reply: "I can't see, but I'm a small AI. I can only read what you type. What are you up to?"

**336.** Typed: "Tell me something cool."

Reply: "The color of the tofu."

**337.** Typed: "What's your favorite animal?"

Reply: "Not really. What's up with you?"

**338.** Typed: "What's your favorite animal?"

Reply: "Not much, what's up with you?"

**339.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a friends."

**340.** Typed: "I'm so bored right now."

Reply: "I know, right?"

**341.** Typed: "I love pizza."

Reply: "Good luck! I hope you have a lovely day."

**342.** Typed: "I'm scared of the dark."

Reply: "What do you mean?"

**343.** Typed: "I'm really tired."

Reply: "That's a really kind of you to happen."

**344.** Typed: "I love pizza."

Reply: "Want to hear a fun fact?"

**345.** Typed: "I'm really tired."

Reply: "That's so kind! I hope you have a nice day."

**346.** Typed: "Why is the sky blue?"

Reply: "I'm an AI called Toby."

**347.** Typed: "Tell me something cool."

Reply: "Here's a silly one. What is the best of you but it can't be fun?"

**348.** Typed: "Are you smarter than me?"

Reply: "Yes, I'm Toby, an AI."

**349.** Typed: "I had a really bad day."

Reply: "What do you mean?"

**350.** Typed: "It's nice to meet you."

Reply: "It's nice to meet you as well! What would you like to talk about?"

**351.** Typed: "Why is the sky blue?"

Reply: "The main language spoken in Cambodia is Khmer."

**352.** Typed: "It's nice to meet you."

Reply: "Oh, I'm glad you're enjoying it."

**353.** Typed: "See you later!"

Reply: "I'm glad you're doing alright. I was just glad you could help."

**354.** Typed: "I got a new puppy."

Reply: "What is this?"

**355.** Typed: "Good night!"

Reply: "Good night! I hope you have lovely dreams."

**356.** Typed: "It's nice to meet you."

Reply: "Yeah, I've been busy lately and busy. I'm glad you're enjoying the day."

**357.** Typed: "Hey, how's your day going?"

Reply: "Not much. Just hanging out with the new weather and stuff."

**358.** Typed: "Can you sing a song?"

Reply: "Sure! Just ask if you want a song, but it's okay to sing."

**359.** Typed: "It's nice to meet you."

Reply: "Nice to meet you, Lisby Lisby Lisby"

**360.** Typed: "I'm really tired."

Reply: "Oh no. Do you want to hear a joke?"

**361.** Typed: "See you later!"

Reply: "Bye! Thanks for chatting with me."

**362.** Typed: "Hey, how's your day going?"

Reply: "Not too bad. Just hanging out. What about you?"

**363.** Typed: "I'm so bored right now."

Reply: "What's the symbol?"

**364.** Typed: "I had a really bad day."

Reply: "Have a great time! Would you like to chat?"

**365.** Typed: "See you later!"

Reply: "Bye! I'm here whenever you want to chat again."

**366.** Typed: "Why is the sky blue?"

Reply: "The blueberry is blue."

**367.** Typed: "Do you ever sleep?"

Reply: "I'm sorry. Would you like a tip for sleeping tonight?"

**368.** Typed: "I'm so bored right now."

Reply: "Worries can feel scary, but it can help you feel calmer."

**369.** Typed: "It's nice to meet you."

Reply: "Thank you! I'm happy to chat with you today."
