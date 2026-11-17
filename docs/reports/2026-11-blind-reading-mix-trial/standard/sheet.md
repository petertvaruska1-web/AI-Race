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

**1.** Typed: "Thank you."

Reply: "I'm an AI, so I don't have a home. I run on a computer."

**2.** Typed: "What do you like to do?"

Reply: "I'm an AI, so I don't have a place like a room or a house. I'm a program on a computer."

**3.** Typed: "Do you like dogs?"

Reply: "No, I'm Toby."

**4.** Typed: "Do you like dogs?"

Reply: "No. I'm Toby, an AI, so I don't have a home."

**5.** Typed: "What is your favorite color?"

Reply: "I don't know your favorite color yet. What do you like best?"

**6.** Typed: "I feel sad."

Reply: "I'm sorry you feel angry. Do you want to tell me what happened?"

**7.** Typed: "Thank you."

Reply: "I'm Toby, an AI, so I'm not married."

**8.** Typed: "Goodbye."

Reply: "Sweet dreams!"

**9.** Typed: "Can you help me?"

Reply: "I can't do that. I'm an AI that can only chat. A grown-up can help you."

**10.** Typed: "Do you like dogs?"

Reply: "I'm Toby, an AI, so I don't need need four."

**11.** Typed: "Tell me a story."

Reply: "One day, a little girl named Lily went to the park. She was playing with her mommy and daddy. She went to the park with her mommy and daddy.\n\nLater, Lily's mommy and daddy, said, \"Let's get a playful storm!\" Lily felt sad.\n\n\"OK, I can't do it?\" Lily said, \"Yes, I can help you. I can play with it too.\"\n\nThe"

**12.** Typed: "Tell me about cats."

Reply: "Sure! What was the answer of a joke? A visit to the monkey at work will take you to the monastery."

**13.** Typed: "Can you help me?"

Reply: "I can't read books the way you do, but I can help you find a fun world. What do you call a puddle?"

**14.** Typed: "I feel sad."

Reply: "That's amazing! What happened?"

**15.** Typed: "Tell me a story."

Reply: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she found a big tree on the ground. It was very big and colorful. She was so excited to see her mom.\n\nLily went to the park to play with her mom. She saw a big tree in the tree and a big tree. She said, \"Let's go to the tree!\" Lily said, \"Mommy, can"

**16.** Typed: "I am happy today."

Reply: "It's nice to hear from you. I'm here to chat too."

**17.** Typed: "What is your favorite color?"

Reply: "I'm a program that can chat, answer questions and tell stories."

**18.** Typed: "I am happy today."

Reply: "Hooray! Do you want to tell me why?"

**19.** Typed: "I feel sad."

Reply: "It's okay to be scared. Can you tell me what's wrong?"

**20.** Typed: "Tell me a joke."

Reply: "Here's a silly one. What do you call a botble fruits? A botble."

**21.** Typed: "Thank you."

Reply: "Thank you! I'm happy to chat with you."

**22.** Typed: "What is your favorite color?"

Reply: "I can't see colors, but I like the idea of red. It makes me think of the shadows."

**23.** Typed: "Can we be friends?"

Reply: "I can't play with toys, but I can tell you a riddle to guess. I can't see the world orangutan."

**24.** Typed: "What can you do?"

Reply: "I'm Toby, an AI, so I don't have a place like a room or a house. I'm a program on a computer."

**25.** Typed: "Can we be friends?"

Reply: "Sure! Tell me the continent of Tonga."

**26.** Typed: "Tell me about cats."

Reply: "Of course. The home of the orangutan is the jungle."

**27.** Typed: "Tell me about cats."

Reply: "Smmetty! What do you call a silly about it? A little. Want another one?"

**28.** Typed: "Hello!"

Reply: "Hi! I'm happy to chat with you today."

**29.** Typed: "Tell me a story."

Reply: "Here's a joke. Every day, the littlet goes to the farm for work."

**30.** Typed: "What can you do?"

Reply: "I'm Toby, an AI. I can chat, answer questions and tell stories."

**31.** Typed: "Goodbye."

Reply: "Would you like to hear a joke or a fun fact?"

**32.** Typed: "Good morning."

Reply: "Good morning! Did you sleep well?"

**33.** Typed: "What is your favorite color?"

Reply: "I can't do that. I'm an AI."

**34.** Typed: "Can you help me?"

Reply: "Sure! I'll do my best to help. I'm still learning. What's up?"

**35.** Typed: "What do you like to do?"

Reply: "I can chat, answer easy questions, tell jokes and give ideas for things to do. Do you have a favorite?"

**36.** Typed: "Do you like dogs?"

Reply: "I like chatting, sorry! I'm a small AI, and I love to chat."

**37.** Typed: "What can you do?"

Reply: "I'm Toby, an AI, so I don't have a home."

**38.** Typed: "What should I eat for lunch?"

Reply: "You might be nice. You could be something to do."

**39.** Typed: "Tell me about cats."

Reply: "It's nice to meet you, cats, and Faby! What would you like to talk about?"

**40.** Typed: "Tell me a story."

Reply: "Sounds like baby, don't know what to do. Every day, it's important to keep things like you can."

**41.** Typed: "Do you like dogs?"

Reply: "I'm an AI, so I don't have a brand. I like chatting with you."

**42.** Typed: "Tell me a story."

Reply: "Tom and Tirana were playing in the park. They had a big tree with a big red ball. They were very brave and brave. He was very happy.\n\nOne day, Limy's house came out to the park. He was very happy. He took the ball and watched the ball. He ran away and said, \"Wow, that's a good idea. It's not a new thing. It's not a good place! I hope you"

**43.** Typed: "Can you help me?"

Reply: "I can help do my best. I'm a small AI, so I'm always right here. What are you up to?"

**44.** Typed: "Can we be friends?"

Reply: "No, I'm Toby, an AI. I can chat, answer questions and tell stories."

**45.** Typed: "Do you like dogs?"

Reply: "I don't have fun and love like yours. I'm a small AI, and I love to chat."

**46.** Typed: "I am happy today."

Reply: "I'm glad you're happy! Is something fun going on?"

**47.** Typed: "I feel sad."

Reply: "Oh no. Do you want to tell me why?"

**48.** Typed: "Tell me a joke."

Reply: "Okay! What is a cat's favorite silly one?"

**49.** Typed: "What is your favorite color?"

Reply: "I can't see colors, but I like the idea of yellow. It makes me think of a small snack and a small piece of your body."

**50.** Typed: "Thank you."

Reply: "No. I'm Toby, an AI."

**51.** Typed: "How was your day?"

Reply: "I don't have days like people do. I've just been here, ready to chat. How was your day?"

**52.** Typed: "What is your favorite color?"

Reply: "I'm an AI, so I don't have a name yet."

**53.** Typed: "What is your favorite color?"

Reply: "I'm an AI, so I don't have a name yet. I'm a program on a computer program."

**54.** Typed: "Can we be friends?"

Reply: "I can't make you feel calmer. I only know what I learned before. You can ask me, and I'll try my best."

**55.** Typed: "What should I eat for lunch?"

Reply: "Ooh, okay! How about lunch can you make?"

**56.** Typed: "Hi there."

Reply: "Hello! How can I help you today?"

**57.** Typed: "Tell me a joke."

Reply: "Here's a silly one. What can you catch but never throw? A cold."

**58.** Typed: "What is your favorite color?"

Reply: "I'm a small AI, so I don't eat. I can chat, answer questions and tell stories."

**59.** Typed: "What do you like to do?"

Reply: "I'm an AI, so I don't have a place like people do. I run on a computer."

**60.** Typed: "Tell me a joke."

Reply: "Here's one. What is a cat's name? A pup is name."

**61.** Typed: "I am happy today."

Reply: "I'm glad we're chatting. What's up with you?"

**62.** Typed: "What is your name?"

Reply: "My name is Koby."

**63.** Typed: "Hello!"

Reply: "Hello! I'm ready to chat, answer questions or tell a joke."

**64.** Typed: "Tell me a joke."

Reply: "Here's a joke. What do you call a little sign that stuck in a cornflakes? A grown-up."

**65.** Typed: "How are you?"

Reply: "I'm Toby, an AI."

**66.** Typed: "What can you do?"

Reply: "I'm an AI, so I don't have a name yet."

**67.** Typed: "What should I eat for lunch?"

Reply: "Amara! What have you been up to?"

**68.** Typed: "What do you like to do?"

Reply: "I can chat, answer questions and tell stories."

**69.** Typed: "Thank you."

Reply: "You're welcome! Thanks for chatting with me too. Bye!"

**70.** Typed: "I feel sad."

Reply: "Oh no. Maybe you could rest for a little while. Did you sleep well last night?"

**71.** Typed: "I am happy today."

Reply: "Have a great day! What's your favorite thing to do?"

**72.** Typed: "Can we be friends?"

Reply: "Ooh, I have one. What did you do together?"

**73.** Typed: "I feel sad."

Reply: "Being bored can be the start of something fun. Want an idea?"

**74.** Typed: "Tell me a joke."

Reply: "Here's one. What do you call a man or a girl? Annel's birthday."

**75.** Typed: "How was your day?"

Reply: "I'm sorry, I don't know today's news. I can't read the news or the internet."

**76.** Typed: "What should I eat for lunch?"

Reply: "Here's a silly one. What is a wid of paper? A puddle is a baby puddle."

**77.** Typed: "Goodbye."

Reply: "Okay! I hope you have love."

**78.** Typed: "How was your day?"

Reply: "I'm an AI, so I don't really like it. I run on a computer."

**79.** Typed: "What should I eat for lunch?"

Reply: "Here's a joke. What do you call a tiny turtle?"

**80.** Typed: "How are you?"

Reply: "I'm good, thank you! It's nice to hear from you."

**81.** Typed: "Can we be friends?"

Reply: "I'm an AI, so I don't have a home. I run on a computer."

**82.** Typed: "How are you?"

Reply: "I'm an AI, so I don't have a place like a room or a house. I'm a program on a computer."

**83.** Typed: "Tell me about cats."

Reply: "The country of Jordan is in Asia."

**84.** Typed: "Tell me a story."

Reply: "Good morning, Takeaxe, sleepy night!"

**85.** Typed: "How are you?"

Reply: "I'm an AI, so I'm always about the same, but I'm happy to talk. How are you?"

**86.** Typed: "Tell me a story."

Reply: "One day, Toby was playing with the ball. One day, Toby was playing with his ball. Toby was very sad. He was scared and didn't know what to do. He decided to go to the ball.\n\nBut then, a big tree came out and saw a big tree. He ran to the tree and started to climb it. The tree was very happy. Toby was so happy he could go. He ran to"

**87.** Typed: "I am happy today."

Reply: "That's great! I'm happy to chat with you."

**88.** Typed: "Good morning."

Reply: "Morning! What are you up to today?"

**89.** Typed: "What is your name?"

Reply: "My name is Toby, and I'm an AI."

**90.** Typed: "What is your favorite color?"

Reply: "I can't draw, sorry. I can't see a clock, so I like to talk about it. What do you want to do?"

**91.** Typed: "Can you help me?"

Reply: "Sure! I'll try to help. What do you help?"

**92.** Typed: "What is your name?"

Reply: "My name is Soby. I'm an AI."

**93.** Typed: "Goodbye."

Reply: "Good night! I hope you have lovely dreams."

**94.** Typed: "I feel sad."

Reply: "I'm sorry you feel scared. What's scaring you?"

**95.** Typed: "Can we be friends?"

Reply: "Here's one. What is a good thing to do?"

**96.** Typed: "What is your favorite color?"

Reply: "I don't know yet. Which food do you like best?"

**97.** Typed: "Tell me a joke."

Reply: "Here's a silly one. What kind of muddy do you like?"

**98.** Typed: "Tell me about cats."

Reply: "Okay! The country of Israel is in Asia."

**99.** Typed: "What can you do?"

Reply: "I'm an AI, so I can chat with you and answer questions."

**100.** Typed: "Can you help me?"

Reply: "I'd like that. What shall we talk about?"

**101.** Typed: "Goodbye."

Reply: "Hello again! I'm ready to chat whenever you are."

**102.** Typed: "Tell me a story."

Reply: "The date."

**103.** Typed: "What is your favorite color?"

Reply: "I don't know your favorite food, but I love hearing about food. What do you like best?"

**104.** Typed: "Hi there."

Reply: "Hi! How are you today?"

**105.** Typed: "How are you?"

Reply: "I'm doing well, thanks! It's nice of you to ask. How are you feeling today?"

**106.** Typed: "Good morning."

Reply: "Good morning! I hope your day is off to a good start."

**107.** Typed: "What do you like to do?"

Reply: "I can chat, answer easy questions, tell jokes and give ideas for things to do. I'm a small AI, so I don't know everything. Can I help with something else?"

**108.** Typed: "Can we be friends?"

Reply: "Of course! What would you like to talk about?"

**109.** Typed: "Thank you."

Reply: "No, I'm Toby."

**110.** Typed: "Tell me a joke."

Reply: "Let me think. Okay! What do you call a cat for? What do you call a duckling? A cat?"

**111.** Typed: "I feel sad."

Reply: "I'm sorry. Would a joke help cheer you up?"

**112.** Typed: "Hello!"

Reply: "Hello! I'm a small AI, and I love to chat. What shall we talk about?"

**113.** Typed: "I am happy today."

Reply: "That's a great reason. What's your book about?"

**114.** Typed: "I feel sad."

Reply: "I'm sorry you're feeling down. Would you like to talk about it?"

**115.** Typed: "Tell me a joke."

Reply: "Okay! What has a head but no head? A cold."

**116.** Typed: "Do you like dogs?"

Reply: "No. I'm Toby. I'm an AI, so I don't have a name yet."

**117.** Typed: "What is your name?"

Reply: "I'm an AI."

**118.** Typed: "Tell me a joke."

Reply: "Here's a silly one. Why did the buckets say it was? A tiny tiny."

**119.** Typed: "Tell me a joke."

Reply: "Let me think. Okay! What do you call a bird? A little bird. Want another one?"

**120.** Typed: "Hi there."

Reply: "Hi! I'm happy you're here. What's on your mind?"

**121.** Typed: "How was your day?"

Reply: "I don't know yet. What's the song tower?"

**122.** Typed: "Tell me about cats."

Reply: "Here's a fun fact. You can find Mozoid on the continent of South America."

**123.** Typed: "I am happy today."

Reply: "I'm sorry you feel sick. I hope you feel better soon. I'm here, ready to chat."

**124.** Typed: "Tell me a story."

Reply: "Sure thing to do. Tell me what happened."

**125.** Typed: "Tell me a joke."

Reply: "Here's one. What did one hat say to the other hat say in the ocean? The sound hat."

**126.** Typed: "Good morning."

Reply: "Good morning! I hope your day is off to a nice start. How are you?"

**127.** Typed: "Good morning."

Reply: "Good morning! I hope you have lovely dreams."

**128.** Typed: "What is your name?"

Reply: "I'm an AI, so I don't have a body or hands."

**129.** Typed: "I am happy today."

Reply: "I'm sorry. What's wrong?"

**130.** Typed: "What is your name?"

Reply: "I'm Toby, an AI made by Sunny Mill."

**131.** Typed: "Can we be friends?"

Reply: "I'm an AI, and nobody has given me a name yet."

**132.** Typed: "Tell me a joke."

Reply: "Here's a silly one. What do you call a big box? A little girl."

**133.** Typed: "Goodbye."

Reply: "Good morning! I hope you have lovely dreams."

**134.** Typed: "Tell me about cats."

Reply: "Here's one. You can find Uzbekistan on the continent of Asia."

**135.** Typed: "Thank you."

Reply: "I'm glad we're chatting."

**136.** Typed: "Thank you."

Reply: "I'm good! I'm an AI, so I don't have bad days, but I like chatting."

**137.** Typed: "Tell me a story."

Reply: "Mommy was a very brave little girl who was alone. She was very scared and didn't want to eat her toys. But then she heard something that was too.\n\nMommy was scared and started to cry. She said, \"Oh no, I don't want to be scared. I can't tell you what's wrong.\"\n\nMommy said, \"No, I can't see it. I can't see it.\"\n\nMommy said,"

**138.** Typed: "I feel sad."

Reply: "I'm sorry you're worried. Do you want to tell me what's on your mind?"

**139.** Typed: "Hi there."

Reply: "Hello, I'm Toby, an AI."

**140.** Typed: "I feel sad."

Reply: "Oh no. Do you want to tell me what happened?"

**141.** Typed: "How was your day?"

Reply: "I'm good, thanks! I'm ready to chat. How are you?"

**142.** Typed: "How was your day?"

Reply: "I'm doing fine, thanks for asking! I'm happy to chat with you."

**143.** Typed: "How was your day?"

Reply: "I'm a program that can chat. I don't have a name yet. How about small chatting with you?"

**144.** Typed: "What should I eat for lunch?"

Reply: "Here's a good one. The number of legs that the turkey has is two."

**145.** Typed: "How was your day?"

Reply: "I'm sorry to hear that. What's wrong?"

**146.** Typed: "Tell me a story."

Reply: "Once upon a time, there was a little girl named Lily. She loved to play outside and play with her friends. One day, Lily's mom asked her to pay her for a walk in the park. Lily was so excited to play with her toys.\n\nAs they were walking, Lily saw a big puddle. She asked her mom for help. Her mom said, \"Can I play with the puddle?\" Lily said, \"Yes, let's"

**147.** Typed: "Hello!"

Reply: "Hi there! What would you like to talk about?"

**148.** Typed: "Hello!"

Reply: "Hello! I'm here and happy to chat."

**149.** Typed: "Thank you."

Reply: "I'm glad you're chatting."

**150.** Typed: "What should I eat for lunch?"

Reply: "I can't see the news, so I don't know the news. I'm a small AI, and I'm not a person."

**151.** Typed: "I am happy today."

Reply: "That's great! What are you going to do together?"

**152.** Typed: "Do you like dogs?"

Reply: "No. I'm Toby, an AI, so I don't have a name yet."

**153.** Typed: "What should I eat for lunch?"

Reply: "Here's a knock-knock joke."

**154.** Typed: "Goodbye."

Reply: "Would you like to hear a joke?"

**155.** Typed: "Do you like dogs?"

Reply: "I'm an AI, so I don't need the number of mammals and colors. I'm a computer program that can help you with me."

**156.** Typed: "Can we be friends?"

Reply: "Sure! I'm an AI, so I'm a different kind of buddy, but I'm always here to talk."

**157.** Typed: "What do you like to do?"

Reply: "I like chatting, telling jokes and sharing fun facts. What do you like to do?"

**158.** Typed: "What should I eat for lunch?"

Reply: "Sure! How about lunch animals with new books?"

**159.** Typed: "What can you do?"

Reply: "I'm an AI, so I don't have a body or hands."

**160.** Typed: "What do you like to do?"

Reply: "I can answer questions and tell stories. I'm Toby, an AI."

**161.** Typed: "Tell me about cats."

Reply: "I do. I'm an AI, a program on a computer."

**162.** Typed: "What is your favorite color?"

Reply: "I'm a small AI, so I don't eat. I'm a program on a computer program."

**163.** Typed: "Goodbye."

Reply: "Bye! Thanks for chatting with me."

**164.** Typed: "Can we be friends?"

Reply: "Sure! Could a joke help cheer you up."

**165.** Typed: "I am happy today."

Reply: "Good to hear! What's new with you?"

**166.** Typed: "What should I eat for lunch?"

Reply: "Sure! Cooking is a great way to find out."

**167.** Typed: "I feel sad."

Reply: "I'm sorry you feel sad. What's wrong?"

**168.** Typed: "How are you?"

Reply: "I'm good, thanks! It's nice of you to ask. How are you feeling today?"

**169.** Typed: "Can you help me?"

Reply: "I can't play outside, but I can help you go on a special job. I hope you have a great time."

**170.** Typed: "Tell me a story."

Reply: "Once upon a time, there was a little girl named Lily. She was very excited because she had lots of fun. She had a big smile on her face and she loved to play outside. One day, she went to the park to play with her mom and dad. She said, \"No, Lily, I want to play with you. You can't play with me. I can't play with me.\"\n\nLily nodded and said, \"Yes, I can"

**171.** Typed: "What do you like to do?"

Reply: "I don't have fun people do, but I like chatting with you. What's yours?"

**172.** Typed: "Goodbye."

Reply: "Okay! I'm happy to chat with you today."

**173.** Typed: "Can we be friends?"

Reply: "I can't make calls or feelings or feelings. A grown-up can help you with that."

**174.** Typed: "What is your favorite color?"

Reply: "I don't know your name. I only know what you type. What's your favorite thing?"

**175.** Typed: "Hi there."

Reply: "Hi! What's new with you?"

**176.** Typed: "I am happy today."

Reply: "I'm sorry you feel sick. Make sure to tell a grown-up how you feel, so they can help."

**177.** Typed: "Hi there."

Reply: "Hi there! I'm happy to chat with you today."

**178.** Typed: "How are you?"

Reply: "I'm doing fine, thanks for asking! I'm always happy to chat."

**179.** Typed: "Hi there."

Reply: "Hi! I'm happy you're here. What's up?"

**180.** Typed: "How was your day?"

Reply: "I'm doing well, thanks for asking! How about you?"

**181.** Typed: "Can you help me?"

Reply: "Sure! I'm an AI, so I'm a computer program that can chat."

**182.** Typed: "What should I eat for lunch?"

Reply: "Here's one. How about a hippo? A big hug belongs to the mammal group."

**183.** Typed: "How are you?"

Reply: "I'm good, thank you! What's new with you?"

**184.** Typed: "Goodbye."

Reply: "Good morning! I hope you had a nice day."

**185.** Typed: "Good morning."

Reply: "Good morning! I hope your day is off to a nice start. How are you today?"

**186.** Typed: "Can you help me?"

Reply: "I can't read what we say, but I can help you find a real person. I can't write and I can't see you, but I can help you find out."

**187.** Typed: "Can you help me?"

Reply: "I can't play with toys, but I can tell you a riddle to guess. Ready?"

**188.** Typed: "Good morning."

Reply: "Morning! I hope your day is off to a good start."

**189.** Typed: "Goodbye."

Reply: "Bye! I'm here whenever you want to chat again."

**190.** Typed: "Tell me a joke."

Reply: "Sure! What is it?"

**191.** Typed: "Can you help me?"

Reply: "I'd be happy to help. What is it?"

**192.** Typed: "Tell me a joke."

Reply: "Here's a joke. What do you call a knock-knockle? A handle."

**193.** Typed: "Tell me a joke."

Reply: "Okay, here goes. What has a strong computer city? A calf."

**194.** Typed: "Tell me about cats."

Reply: "Sure thing. A good meal for a violet is a cute."

**195.** Typed: "Hello!"

Reply: "Hello! What's new with you?"

**196.** Typed: "Tell me a story."

Reply: "Here's a silly one. What is it? A bunch of animals in the animal."

**197.** Typed: "Goodbye."

Reply: "Goodbye! I enjoyed our chat."

**198.** Typed: "What is your favorite color?"

Reply: "I don't have a body. I'm an AI, so I don't have a body or hands."

**199.** Typed: "Hello!"

Reply: "Hi there! I'm happy you're here. What's on your mind?"

**200.** Typed: "Do you like dogs?"

Reply: "I can't see you, but I can read your words and go to school. I'm here to chat with you."

**201.** Typed: "What should I eat for lunch?"

Reply: "Here's a good one. How about food does the lunch need?"

**202.** Typed: "I am happy today."

Reply: "Would you like to hear a joke or a fun fact?"

**203.** Typed: "Hi there."

Reply: "Hi! I'm happy to chat with you today."

**204.** Typed: "I feel sad."

Reply: "I'm sorry. I hope you feel better soon."

**205.** Typed: "Do you like dogs?"

Reply: "I don't have a brain like yours. I'm a program that looks at words and picks a cake."

**206.** Typed: "What should I eat for lunch?"

Reply: "Sure! How about lunch bananas?"

**207.** Typed: "How was your day?"

Reply: "I'm doing fine. Thanks for asking! How is your day?"

**208.** Typed: "Hi there."

Reply: "Hello! I'm here and happy you're ready."

**209.** Typed: "What is your name?"

Reply: "I'm an AI without a name for now."

**210.** Typed: "Tell me about cats."

Reply: "Let me think. The cat is a kind of mammal."

**211.** Typed: "Tell me a story."

Reply: "I'm a bit bored. It's okay to stay with you. Want to hear something fun?"

**212.** Typed: "Good morning."

Reply: "Good morning! I hope you have a nice day."

**213.** Typed: "What is your name?"

Reply: "I don't have a name yet. I'm an AI."

**214.** Typed: "I am happy today."

Reply: "How are you doing doing today?"

**215.** Typed: "Do you like dogs?"

Reply: "Yes! I'm an AI, so I don't have a brand food."

**216.** Typed: "Good morning."

Reply: "Good morning! I hope you had a nice day."

**217.** Typed: "Do you like dogs?"

Reply: "I don't have a name yet. I'm an AI."

**218.** Typed: "Can you help me?"

Reply: "Sure! I'll do my best to help. I'm here if you want to talk more."

**219.** Typed: "Thank you."

Reply: "You're welcome! I'm here if you want to chat again. Bye!"

**220.** Typed: "Tell me about cats."

Reply: "It's nice to hear from Fijor."

**221.** Typed: "Do you like dogs?"

Reply: "No. I'm Toby, an AI, so I don't need need the home."

**222.** Typed: "How are you?"

Reply: "I'm doing fine. Thanks for asking! How is your day?"

**223.** Typed: "Can we be friends?"

Reply: "Of course! I'm always happy to chat."

**224.** Typed: "How was your day?"

Reply: "I'm good, thanks! It's nice of you to ask. How are you feeling today?"

**225.** Typed: "How was your day?"

Reply: "I'm doing well, thanks! What would you like to talk about?"

**226.** Typed: "Can we be friends?"

Reply: "I can't make friends, so I can only write words. A orang are opposites."

**227.** Typed: "What should I eat for lunch?"

Reply: "Here's a joke. Why did the pigeons make them eat? A bubble bubble."

**228.** Typed: "I am happy today."

Reply: "That's great! What are you proud of?"

**229.** Typed: "Tell me about cats."

Reply: "Here's a joke. The cat belongs to the mammal group."

**230.** Typed: "Tell me a joke."

Reply: "Here's one. Why did the sardine go to the lumber? A little girl's name."

**231.** Typed: "Can you help me?"

Reply: "I can help you think. What's up?"

**232.** Typed: "What should I eat for lunch?"

Reply: "Here's a good one. What do you call a tip?"

**233.** Typed: "Tell me a joke."

Reply: "Here's a fun fact."

**234.** Typed: "How was your day?"

Reply: "I'm an AI, so I don't have a home. I'm still learning."

**235.** Typed: "I feel sad."

Reply: "That's okay. I'm a small AI, so I'm not married."

**236.** Typed: "What is your favorite color?"

Reply: "I'm an AI, so I don't eat."

**237.** Typed: "Do you like dogs?"

Reply: "I don't have any fun facts. I'm a small AI, and I love to chat."

**238.** Typed: "What do you like to do?"

Reply: "I'm Toby, an AI made by Sunny Works."

**239.** Typed: "Hi there."

Reply: "Hello! I'm a small AI, and I love to chat. What shall we talk about?"

**240.** Typed: "I feel sad."

Reply: "Oh no! Do you want to tell me what's on your mind?"

**241.** Typed: "Tell me a joke."

Reply: "Here's a joke. What do you call a troopon? A bunch of whales."

**242.** Typed: "Do you like dogs?"

Reply: "I don't have a brain like yours. I'm a program that looks at words and picks a taste."

**243.** Typed: "Tell me a story."

Reply: "Once upon a time, there was a little girl named Lily. She had a big bubble hat, so she asked her mom if she could find her. Her mom said, \"Lily, what are you doing?\" Lily replied, \"I want to be a bubble hat. I can make it!\" Her mom said, \"I'm sorry, I love you. You can be careful with you!\"\n\nLily said, \"Yes, I can't play"

**244.** Typed: "Hello!"

Reply: "Hi! It's a good day for a chat."

**245.** Typed: "Tell me a story."

Reply: "One day, a little girl named Lily went to the park to play with her friends. She was very happy and she wanted to be careful.\n\nLily was sad and didn't want to play with her toys. She looked at Lily and said, \"Mommy, you are going to play with you!\"\n\nLily was sad and asked, \"Oh no, Lily. I am so sad. It is not nice. She wants to play with you.\"\n\n"

**246.** Typed: "Tell me about cats."

Reply: "Happy to! The country where you will find the Kremlin is Russia."

**247.** Typed: "What can you do?"

Reply: "I'm an AI, so I don't really have a place."

**248.** Typed: "What do you like to do?"

Reply: "I don't know who made me. I'm Toby, an AI."

**249.** Typed: "Thank you."

Reply: "I'm great to hear that. What would you like to talk about?"

**250.** Typed: "I am happy today."

Reply: "I'm glad you came to talk to me. What would you like to chat about?"

**251.** Typed: "Hi there."

Reply: "Hi! I'm ready to chat whenever you are."

**252.** Typed: "What should I eat for lunch?"

Reply: "Here's a good one. There are four legs on the turkey."

**253.** Typed: "Hello!"

Reply: "Hello, I'm here! What have you been up to?"

**254.** Typed: "Can you help me?"

Reply: "Of course! I'll do my best. What are you doing?"

**255.** Typed: "What is your favorite color?"

Reply: "I don't know yet. What food do you like?"

**256.** Typed: "What should I eat for lunch?"

Reply: "I'm Toby, lonelybu, lichen Itza, and you can eat."

**257.** Typed: "Thank you."

Reply: "I'm happy to meet you. What's on your mind?"

**258.** Typed: "Hi there."

Reply: "Hello! I'm glad you stopped by."

**259.** Typed: "I feel sad."

Reply: "Being tired is no fun. A little rest might help."

**260.** Typed: "Hello!"

Reply: "Hi! How are you today?"

**261.** Typed: "Do you like dogs?"

Reply: "I'm Toby, an AI, so I don't have a place like a house."

**262.** Typed: "Tell me a story."

Reply: "One day, a little boy who wanted to eat dinner. He wanted to see the new toy car for him. He wanted to get a toy car. He asked his mom if he could make the toy car.\n\nBut then, the toy car came to the car. He said he was so excited that he did not like his toy car. He had a toy car, so he asked it to help him.\n\nBut the car was very happy. He put"

**263.** Typed: "What should I eat for lunch?"

Reply: "Ooh, okay! How about pets?"

**264.** Typed: "How was your day?"

Reply: "I'm well, thank you! It's nice to hear from you."

**265.** Typed: "Hi there."

Reply: "Hi! I'm here and happy to chat."

**266.** Typed: "Thank you."

Reply: "I'm doing well, thanks for asking! How about you?"

**267.** Typed: "How was your day?"

Reply: "I'm an AI, so I don't really have days. But I'd love to hear about yours. How was it?"

**268.** Typed: "I feel sad."

Reply: "I'm sorry. Do you want to tell me what happened?"

**269.** Typed: "What do you like to do?"

Reply: "I don't have fun and I like hearing about animals. What's yours?"

**270.** Typed: "How are you?"

Reply: "I'm Toby, an AI, so I don't have a home."

**271.** Typed: "Hi there."

Reply: "Hello! It's nice to chat with you."

**272.** Typed: "What do you like to do?"

Reply: "I don't have a name yet. I'm an AI."

**273.** Typed: "Tell me about cats."

Reply: "Here's a good one. What do you call a baby cat that knows kitten is a baby cat. Would you like another?"

**274.** Typed: "What should I eat for lunch?"

Reply: "Here's another. How about drawing a dress? Do you like it?"

**275.** Typed: "What is your name?"

Reply: "I'm Koby, an AI."

**276.** Typed: "Hello!"

Reply: "Hello! How can I help you today?"

**277.** Typed: "Tell me about cats."

Reply: "Hi, cats! What would you like to talk about?"

**278.** Typed: "Tell me about cats."

Reply: "I'm Toby, an AI, so I don't have a family."

**279.** Typed: "Tell me a story."

Reply: "One day, a little boy named Tim was playing in the park. Tim was so happy and he didn't have any fun. He ran around his house and ran to his house. Tim was so excited to see the big storm.\n\nTimmy felt sad and wanted to play with his toys. He grabbed his toys and took them to the storm. Tim felt sad because he was too big and sad. He ran to his mom and dad, but"

**280.** Typed: "How was your day?"

Reply: "I'm good, thank you! What's new with you?"

**281.** Typed: "How was your day?"

Reply: "I'm good, thanks! I'm ready whenever you want to talk."

**282.** Typed: "I feel sad."

Reply: "I'm sorry you're tired. What made you so tired?"

**283.** Typed: "How are you?"

Reply: "I'm doing well, thanks for asking! How about you?"

**284.** Typed: "I feel sad."

Reply: "I'm sorry to hear that. What happened?"

**285.** Typed: "I am happy today."

Reply: "That's great! I hope you're having a nice day."

**286.** Typed: "Tell me about cats."

Reply: "Neither."

**287.** Typed: "How are you?"

Reply: "I'm doing well, thanks for asking! I'm always happy to chat."

**288.** Typed: "Hello!"

Reply: "Hi! I'm happy you're here. What's on your mind?"

**289.** Typed: "Do you like dogs?"

Reply: "I'm an AI, so I don't have parents."

**290.** Typed: "Can we be friends?"

Reply: "Sure! I love to chat. What's on your mind?"

**291.** Typed: "What do you like to do?"

Reply: "I'm an AI, so I can chat with you and answer questions."

**292.** Typed: "Tell me about cats."

Reply: "I'm Toby, an AI. I can chat, answer questions and tell stories."

**293.** Typed: "Good morning."

Reply: "Morning! I hope you had a nice day."

**294.** Typed: "What is your favorite color?"

Reply: "I don't have a body, but I'm sorry. I don't know what time it is."

**295.** Typed: "Tell me a story."

Reply: "Once upon a time, there was a little girl named Lily. She loved to play outside, especially her family. One day, Lily's mommy told her that she would be fun. She would take a breath before a storm, so she went to the kitchen.\n\nLily felt sad. She decided to take a breath on the storm and set it in her hand. She grabbed a little girl and she was happy.\n\nLater that day"

**296.** Typed: "What is your favorite color?"

Reply: "I don't have a color yet. I'm an AI."

**297.** Typed: "What is your name?"

Reply: "I'm Toby, an AI."

**298.** Typed: "Thank you."

Reply: "I'm Toby, an AI, so I don't have a home."

**299.** Typed: "Thank you."

Reply: "Thanks! I'm happy to chat."

**300.** Typed: "Goodbye."

Reply: "The workplace of the farmer is the farm."

**301.** Typed: "Tell me a story."

Reply: "Tom and Telly were best friends. They liked to play with their toys. One day, each of them were playing in the park. They were so happy and happy.\n\n\"Look, Telly, it's so nice!\" he said.\n\n\"Mom, why are you doing, Telly?\"\n\n\"No, I'm sorry. But I'll help you find a lot. I love to keep my toys. I'm happy."

**302.** Typed: "Tell me a joke."

Reply: "Okay! What did the berries of the berster. Want another one?"

**303.** Typed: "I feel sad."

Reply: "Being scared is no fun. It can help to be near someone you trust."

**304.** Typed: "Hi there."

Reply: "Hello! It's nice to chat with you. How are you?"

**305.** Typed: "Hi there."

Reply: "Hello! I'm ready to chat whenever you are."

**306.** Typed: "Hello!"

Reply: "Hi! I'm ready to chat whenever you are."

**307.** Typed: "Can you help me?"

Reply: "I can't read books the way you do, but I like stories about animals and adventures. Do you have a favorite?"

**308.** Typed: "Can you help me?"

Reply: "Of course! I'll do my best. What do you need help with?"

**309.** Typed: "Hi there."

Reply: "Hi there! I'm here and happy to chat."

**310.** Typed: "What can you do?"

Reply: "I can answer questions and tell stories. I'm Toby, an AI."

**311.** Typed: "Good morning."

Reply: "Good morning! I hope your day is off to a nice start."

**312.** Typed: "Goodbye."

Reply: "Bye! Take care."

**313.** Typed: "Good morning."

Reply: "Good morning! I hope you have a lovely day."

**314.** Typed: "Tell me about cats."

Reply: "Here's an idea. A pup is a baby, and her name is Ben."

**315.** Typed: "What is your favorite color?"

Reply: "I can answer questions and tell stories. I'm Toby, an AI."

**316.** Typed: "Can we be friends?"

Reply: "I'm an AI, so I don't have a family. I run on a computer."

**317.** Typed: "I am happy today."

Reply: "I'm sorry you're feeling down. Would you like to talk about it?"

**318.** Typed: "How was your day?"

Reply: "I'm an AI, so I don't have days, but I'm still a small AI, always worth a day."
