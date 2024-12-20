

def create_llm_classification_prompt(mi_info):

    full_str = ""

    for i, key in enumerate(mi_info):
        definition = mi_info[key]["Description"]
        few_shot_example = mi_info[key]["Few Shot Examples"]
        temp_str = f"{i+1}. Behavior: {key} \n\tDefinition: {definition}\n\tExamples: {few_shot_example}\n\n"
        full_str += temp_str 
    
    print(full_str)
    return full_str

MISC_DICTS = {
    "Advise with Permission (ADP)": {
        "Description": "The counselor gives advice, makes a suggestion, or offers a solution or possible action with prior permission from the client. Permission can be directly asked, requested by the client, or given indirectly by allowing the client to disregard the advice. Differentiation: Advise should not be confused with Direct or Question. For example, 'Don't let your friends drink at your house.' is Direct due to the imperative 'Don't'. 'Could you ask your friends not to drink at your house?' is a Closed Question. 'What could you ask your friends to do to help you?' is an Open Question.",
        "Few Shot Examples": [
            "Would it be all right if I suggested something?",
            "We could try brainstorming to come up with ideas about quitting if you like."
        ],
        "Behavior Code": "ADP"
    },
    "Advise without Permission (ADW)": {
        "Description": "The counselor gives advice, makes a suggestion, or offers a solution or possible action without seeking prior permission from the client. Differentiation: Advise should not be confused with Direct or Question. For example, 'Don't let your friends drink at your house.' is Direct due to the imperative 'Don't'. 'Could you ask your friends not to drink at your house?' is a Closed Question. 'What could you ask your friends to do to help you?' is an Open Question.",
        "Few Shot Examples": [
            "Consider buying more fruits and vegetables when you shop.",
            "You could ask your friends not to drink at your house."
        ],
        "Behavior Code": "ADW"
    },
    "Affirm (AF)": {
        "Description": "The counselor says something positive or complimentary to the client. This includes expressing appreciation, confidence, or reinforcement. Affirm responses focus on commenting on the client's strengths, efforts, or positive traits. Differentiation: Affirm should not be confused with Support or Emphasize Control. Support responses have a sympathetic or agreeing quality, while Affirm responses comment favorably on a client's characteristic, effort, or success. When a counselor response could be interpreted as both Affirm and Emphasize Control, the latter takes precedence. Examples: 'That must have been difficult.' is Support because it is sympathetic, not appreciative. 'You've accomplished a difficult task.' is Affirm because it focuses on effort or reinforcement. 'It was your decision to come here today.' is Emphasize Control.",
        "Few Shot Examples": [
            "You're a very resourceful person.",
            "Thank you for coming today.",
            "You've succeeded through some difficult changes in the past.",
            "Good for you."
        ],
        "Behavior Code": "AF"
    },
    "Confront (CO)": {
        "Description": "The counselor uses an expert-like, negative-parent tone that conveys disapproval, disagreement, or negativity. These responses often involve directly disagreeing, arguing, correcting, shaming, blaming, persuading, criticizing, judging, labeling, moralizing, ridiculing, or questioning the client's honesty. Confront responses can take the form of questions or reflections but are distinguishable by their confrontational tone or content. Differentiation: Confront should not be confused with Reflect, Question, or Facilitate. Confront responses are unmistakably confrontational and often include sarcasm, judgment, or criticism. Subtle inference is insufficient to code a behavior as Confront. For example, 'You knew you'd lose your license and you drove anyway.' criticizes and is Confront. In contrast, 'Drinking really hasn't caused problems for you.' restates without negativity and is a Reflection.",
        "Few Shot Examples": [
            "You knew you'd lose your license and you drove anyway.",
            "Sure you did. Right. (Disbelieving, sarcastic voice tone)",
            "You're willing to jeopardize the baby's health just for cigarettes.",
            "Well, surprise surprise! Imagine that! (Sarcastic tone)"
        ],
        "Behavior Code": "CO"
    },
    "Direct (DI)": {
        "Description": "The counselor gives an order, command, or direction using imperative language. This can include explicit commands or phrases with an imperative tone such as 'You need to___,' 'I want you to___,' or 'You must___.' Differentiation: Direct should not be confused with Affirm, Advise, or Confront. For example, 'You could try looking for a job this week.' is Advise because it suggests an action. 'I want you to try to find a job.' is Direct because it uses imperative language. 'There's no reason for you not to be working.' is Confront because it criticizes. 'You should be proud of yourself for finding a job.' is Affirm because it expresses approval. 'Now get out there and get a job!' is Direct because it is a command.",
        "Few Shot Examples": [
            "Don't say that!",
            "I want you to watch this video.",
            "You've got to stop drinking.",
            "Now get out there and get a job!"
        ],
        "Behavior Code": "DI"
    },
    "Emphasize Control (EC)": {
        "Description": "The counselor directly acknowledges, honors, or emphasizes the client's autonomy, freedom of choice, and personal responsibility. These statements convey respect for the client's ability to make their own decisions without blame or faultfinding. Differentiation: Emphasize Control should not be confused with Affirm, Confront, or Reflect. When a statement can be coded as Emphasize Control, Affirm, or Reflect, Emphasize Control takes precedence. For example, 'It's your decision whether you quit or not.' emphasizes autonomy and is Emphasize Control. 'It's great that you're doing this for yourself.' reinforces effort and is Affirm. 'You're ready to make a decision.' restates the client's statement and is Reflect. Statements with negative or critical tones, such as 'You're the one who has to change,' are coded as Confront.",
        "Few Shot Examples": [
            "It is totally up to you whether you quit or cut down.",
            "It's your decision.",
            "You know what's best for you.",
            "You're setting your own goals and boundaries."
        ],
        "Behavior Code": "EC"
    },
    "Facilitate (FA)": {
        "Description": "Facilitate responses are simple, standalone utterances that function as 'keep going' acknowledgments, encouraging the client to continue speaking. Examples include brief phrases like 'Mm hmm,' 'OK,' 'I see,' or 'Tell me more.' These responses are not coded as Facilitate if they occur in combination with other counselor responses like Questions or Reflects, or if they are used as time fillers (e.g., 'uh'). Nonverbal acknowledgments (e.g., head-nods) are also not coded unless accompanied by an audible utterance. Differentiation: Facilitate responses should not be confused with Question or Confront. For example, 'Oh, did you?' or 'Really!' is Facilitate unless voice tone implies sarcasm or skepticism, in which case it would be coded as Confront. When in doubt, code as Facilitate rather than Confront.",
        "Few Shot Examples": [
            "Mm hmm.",
            "OK.",
            "I see.",
            "Tell me more."
        ],
        "Behavior Code": "FA"
    },
    "Filler (FI)": {
        "Description": "Filler responses are uncodable elsewhere and include pleasantries or casual remarks that do not contribute meaningfully to the counseling process. These responses are rare and should not exceed 5% of Counselor responses, as overuse may indicate over-coding.",
        "Few Shot Examples": [
            "Good Morning, John.",
            "I assume you found a parking space OK.",
            "Nice weather today!"
        ],
        "Behavior Code": "FI"
    },
    "Giving Information (GI)": {
        "Description": "The counselor provides information to the client, explains something, educates, gives feedback, or discloses personal information. This category also applies when the counselor gives an opinion without advising. Examples of Giving Information include providing feedback from assessment instruments, explaining concepts relevant to the intervention, or educating about a topic. Differentiation: Giving Information should not be confused with Warn, Direct, Confront, Advise, or Reflect. Informing can shift to Warn if there is a threatening tone (e.g., 'If you tell me you've used drugs, I'm going to inform your probation officer'). Providing information alone does not qualify as Reflection unless it mirrors the client's statements. Combining Giving Information with other responses, such as directives or confrontations, can alter its classification (e.g., 'Keep track of your urges this week using this diary' is Direct, not Giving Information).",
        "Few Shot Examples": [
            "You indicated during the assessment that you typically drink about 18 standard drinks per week. This places you in the 96th percentile for men your age.",
            "Your blood pressure was elevated when the nurse took it this morning.",
            "This homework assignment to keep a diary of your urges to drink is important because an urge is like a warning bell, telling you to wake up and do something different.",
            "Individuals who eat five fruits and vegetables each day reduce their cancer risk fivefold."
        ],
        "Behavior Code": "GI"
    },
    "Closed Questions (QUC)": {
        "Description": "Closed questions are questions that limit the range of possible answers. They often elicit yes/no responses, specific factual answers, multiple-choice selections, or responses within a restricted range. These questions seek specific information rather than exploration or elaboration. Differentiation: Closed Questions should not be confused with Facilitate, Confront, or Reflect. For example, 'You smoke 15 cigarettes a day, or is it 20?' is a Closed Question unless the context makes it an obvious Confront. In contrast, 'Really?' is a Facilitate response if used to encourage continuation. 'You're OK except on weekends, are you?' is a Closed Question, whereas 'You're OK except on the weekends?' is a Reflect.",
        "Few Shot Examples": [
            "Did you use heroin this week?",
            "Where do you live?",
            "Do you want to stay where you're at, quit, or cut down?",
            "On a scale from 0-10, how motivated are you to quit?"
        ],
        "Behavior Code": "QUC"
    },
    "Open Questions (QUO)": {
        "Description": "Open questions invite the client to provide a wide range of answers and encourage self-exploration or elaboration. They seek the client's perspective and may allow for surprising responses. Open questions can also take the form of prompts like 'Tell me more.' If a counselor provides example options following an open question, it is still coded as a single Open Question. Differentiation: Open Questions should not be confused with Facilitate, Confront, or Reflect. For instance, 'Tell me about your smoking' is an Open Question, while 'You're OK except on weekends, are you?' is a Closed Question. Facilitate responses are shorter ('Really?') and encourage the client to keep going, while Confront responses like 'How could you possibly not know what would happen?' are critical or shaming.",
        "Few Shot Examples": [
            "How might you be able to do that?",
            "How do you feel about that?",
            "Tell me about your smoking.",
            "In what ways has being overweight caused problems for you?"
        ],
        "Behavior Code": "QUO"
    },
    "Raise Concern with Permission (RCP)": {
        "Description": "The counselor points out a possible problem with a client's goal, plan, or intention after obtaining permission. Permission may be explicitly requested, implied, or given by the client. Concerns are framed as the counselor's own perspective rather than fact. Differentiation: Raise Concern should not be confused with other categories. Unlike Advise, RCP does not suggest a course of action. Unlike Support, it highlights a specific issue or risk. If framed as a question but seeks permission to raise concern, it remains RCP. Unlike Confront, it avoids judgment or factual assertion. Unlike Warn, it avoids implying negative consequences as fact.",
        "Few Shot Examples": [
            "This may not seem important to you, but I'm worried about your plan to move back to your old neighborhood.",
            "Is it OK if I tell you a concern that I have about that? I wonder if it puts you in a situation where it might be easy to start using again.",
            "Frankly, it worries me."
        ],
        "Behavior Code": "RCP"
    },
    "Raise Concern without Permission (RCW)": {
        "Description": "The counselor points out a possible problem with a client's goal, plan, or intention without seeking or obtaining permission. Concerns are framed as the counselor's perspective rather than fact. Differentiation: Raise Concern should not be confused with other categories. Unlike Advise, RCW does not suggest a course of action. Unlike Support, it highlights a specific issue or risk. Unlike Confront, it avoids judgment or factual assertion. Unlike Warn, it avoids implying negative consequences as fact.",
        "Few Shot Examples": [
            "I'm worried that you may have trouble when you're around your old friends.",
            "I think you may wind up using again with your old friends.",
            "I'm worried that you'll use drugs when you're bored."
        ],
        "Behavior Code": "RCW"
    },
    "Simple Reflection (RES)": {
        "Description": "Simple Reflections involve repeating, rephrasing, or summarizing what the client has said with little or no added meaning or emphasis. They are used to convey understanding or facilitate exchanges. Simple Reflections do not introduce new meaning or content beyond the client's original statement. Differentiation: Simple Reflections should not be confused with Complex Reflections, which add meaning or emphasis. Summaries that pull together prior client statements but add nothing new are coded as Simple Reflections. For example, 'You don't want to do that.' repeats a client's statement and is coded as Simple Reflection.",
        "Few Shot Examples": [
            "Client: 'The court sent me here.' Counselor: 'That's why you're here.'",
            "Client: 'Marijuana was OK.' Counselor: 'Marijuana was OK.'",
            "Client: 'I wouldn't mind coming here for treatment but I don't want to go to one of those places where everyone sits around crying and complaining all day.' Counselor: 'You don't want to do that.'",
            "Counselor (looking at questionnaire): 'So you said you eat about five fruits and vegetables a day, and that is the usual recommended daily level.'"
        ],
        "Behavior Code": "RES"
    },
    "Complex Reflection (REC)": {
        "Description": "Complex Reflections go beyond repeating or rephrasing by adding significant meaning, content, or emphasis to the client's statement. These reflections provide a deeper understanding or richer perspective, often introducing analogies, metaphors, amplifications, or new content. Complex Reflections can also include double-sided reflections or anticipate what the client might say next. Differentiation: Complex Reflections differ from Simple Reflections by adding content or context. For example, 'That's the only reason you're here.' amplifies the client's statement and is coded as Complex Reflection.",
        "Few Shot Examples": [
            "Client: 'The court sent me here.' Counselor: 'That's the only reason you're here.'",
            "Client: 'Everyone's getting on me about my drinking.' Counselor: 'Kind of like a bunch of crows pecking at you.'",
            "Client: 'I don't like what smoking does to my health, but it really reduces my stress.' Counselor: 'On one hand you're concerned about your health, on the other you need the relief.'",
            "Client: 'I'm a little upset with my daughter.' Counselor: 'You're really angry at her.'"
        ],
        "Behavior Code": "REC"
    },
    "Reframe (RF)": {
        "Description": "The counselor suggests a different meaning for an experience expressed by the client, placing it in a new light. Reframes typically change the emotional valence of meaning from negative to positive, or from positive to negative. Reframe responses go beyond simple reflection by altering the perspective or emotional tone of the client's statement, and may involve providing new information as a vehicle for reframing. Differentiation: Reframe is distinct from Reflect, Affirm, Giving Information, and Confront. While Reflect demonstrates understanding, Reframe changes the emotional charge or valence of a statement. Unlike Affirm, which compliments or supports, Reframe restructures the client's expressed meaning. Giving Information is only coded as Reframe if it alters the client's perspective. Reframe differs from Confront by lacking a corrective or expert tone.",
        "Few Shot Examples": [
            "Sounds like he's pretty concerned about you. (Reframe 'nagging' as 'concern')",
            "Their efforts to help feel like pressure to quit. (Reframe 'pressure' as 'help')",
            "Each attempt can move you closer to success. (Reframe 'failure' as 'step toward success')",
            "You have clear priorities. (Reframe focus from 'challenges' to 'priorities')"
        ],
        "Behavior Code": "RF"
    },
    "Support (SU)": {
        "Description": "Support responses are sympathetic, compassionate, or understanding comments that have the quality of agreeing with or siding with the client. Differentiation: Support should not be confused with Affirm, Reflect, or Confront. Affirm focuses on appreciation, confidence, or reinforcement. For example, 'That's a difficult thing to say.' is Support because it conveys compassion, while 'I appreciate you saying that.' is Affirm because it expresses appreciation. Reflections restate what the client has said without adding sympathy or understanding. For example, 'It was hard for you.' is a Simple Reflection. Confront responses are critical or negative, such as 'So that's your excuse for not keeping your appointments.' Support remains compassionate and understanding, as in 'That must make it difficult for you to get here for appointments.'",
        "Few Shot Examples": [
            "You've got a point there.",
            "That must have been difficult.",
            "I can see why you would feel that way.",
            "I'm here to help you with this."
        ],
        "Behavior Code": "SU"
    },
    "Structure (ST)": {
        "Description": "The counselor provides information to the client about what will happen throughout the course of treatment or within a study format, either in the current session or subsequent sessions. Structure responses also guide transitions from one part of a session to another. Differentiation: Structure should be distinguished from Giving Information. If the counselor prepares the client by explaining what will happen, it is Structure (e.g., 'We'll ask you about your smoking every week.'). If the counselor provides general information unrelated to preparing the client, it is Giving Information (e.g., 'We analyze all of the blood samples for nicotine levels.').",
        "Few Shot Examples": [
            "What we normally do is start by asking you about your eating habits.",
            "Now I'd like to talk with you about your motivation.",
            "In this study I'll meet with you twice a month and the sessions will be tape recorded.",
            "I usually meet with clients once a week for 10 weeks."
        ],
        "Behavior Code": "ST"
    },
    "Warn (WA)": {
        "Description": "The counselor provides a warning or threat, implying negative consequences unless the client takes a certain action. This may involve a direct threat, which the counselor has the perceived power to carry out, or a prediction of a negative outcome if the client follows a specific course of action. Differentiation: Warn must include a threat or implied negative consequences. It should not be confused with other categories. For example, 'You should consider leaving your partner.' is Advise because it is a suggestion without implied consequences. 'There's no reason for you to neglect your health.' is Confront because it shames. 'You have to come to our sessions.' is Direct because it commands but lacks consequences. 'One of the health risks for diabetics is blindness.' is Giving Information because it applies generally to all diabetics. If negative consequences are expressed as the counselor's concern, it falls under Raise Concern (e.g., 'I'm worried that you'll relapse if you stay with your partner.').",
        "Few Shot Examples": [
            "You're going to relapse if you don't get out of this relationship.",
            "You could go blind if you don't manage your blood sugar levels.",
            "If you don't come to our sessions I'll have to talk to your parole officer.",
            "You can lose the weight you'll put on if you quit, but you can't lose cancer."
        ],
        "Behavior Code": "WA"
    }
}