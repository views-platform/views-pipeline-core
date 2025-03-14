# Model-Specific Input Data Querysets

| ADR Info            | Details                             |
|---------------------|-------------------------------------|
| Subject             | Model-Specific Input Data Querysets |
| ADR Number          | 006                                 |
| Status              | Proposed                            |
| Author              | Jim                                 |
| Date                | 27.02.2025                          |

## Context
Most VIEWS models will need to fetch input data (raw or transformed) from the central VIEWS database on gjoll, custom-built for the project, curated and protected by a VPN and SSL certificates. A custom-built transformation layer allows raw data to be transformed in a wide variety of fashions suitable for VIEWS regression-based models.

Data is fetched from the database using a custom querying object referred to as a Queryset. A Queryset is a representation of a multi-column dataset with a common (time-unit, spatial-unit) index, allowing users to fetch raw data and specify an arbitrary chain of transforms to be applied to any feature, before returning a single compressed dataframe (or, in future, tensor).

Fetching a **new** queryset is a two-step process. Before a new queryset can be fetched via its .fetch() method, it must first be published via its .publish() method. This operation stores the **definition** of the queryset (not its data) in a custom database on gjoll. Querysets that have not been published cannot be fetched.

The rationale behind this functionality was to provide standard querysets which, once published by one user, could be fetched by anyone **without needing the queryset's definition** - querysets can be fetched by name from the database.

In practice, this functionality is almost never used - users almost always have the queryset definition coded and simply chain the .publish().fetch() methods. The advantage of doing this is that users can always be confident that they know exactly what they are fetching, and that any updates to their queryset are made before fetching it. Fetching a queryset by name may give unexpected results if the queryset definition has been updated by another user in the meantime (such events have caused problems in the past).

Other alternatives that could be considered are
- making all querysets read-only once they are published, so that new versions must be given different names (queryset versioning, essentially)
- defining some querysets as 'protected' so that only some users can update them (other users would have to make something resembling a pull request)

Both these options would strongly reduce the flexibility of the data querying infrastructure however.

## Decision
It was therefore decided that 
- every VIEWS model needing to fetch data from the central database should have its own queryset defined in a `config_queryset.py` file in the model's `/configs/` directory.
- data for every queryset is fetched by chaining the .publish().fetch() methods, so that whatever queryset definition is present at runtime is pushed to the database and immediately fetched 

QUESTION: What is **not** enforced at present is that model querysets have unique names - several models could have querysets with the same name, and in principle with **different definitions**. When fetching querysets in series as we are now, this does not really matter very much - models will all get the data they want because we are always doing queryset.publish().fetch(). However, if we fetch querysets in parallel, or if two users happen to fetch querysets with the same name but different definitions **at the same time**, we could experience data races and undefined behaviour - while a queryset fetch is **actually in progress**, the server refers to querysets by name. If two querysets with the same name but different definitions come in just after each other, the server will never check the definition of the second queryset, since it thinks it is already working on a queryset with that name.
The best solution would be to implement a fix server-side that treats such qs's as distinct entities. 
In the meantime, to reduce the risk of data races occurring, querysets could simply be named after the model they belong to. This arguably makes more sense anyway - if are not centrally defining querysets, surely they should just be named after the model they belong to? Should this be enforced?

## Consequences
**Positive Effects**:
- modellers can easily locate the definition of the queryset for any model. Querysets are reasonably easy to read, so a good idea of what data goes into any model can be quickly obtained.
- as long as the unlikely happenstance that two or more users are trying to fetch different querysets with the same name at the same time does not eventuate, modellers can be certain that they will get the data requested, since performing a .publish().fetch() chain updates the queryset definition immediately before fetching it.

**Negative effects**:
- this potentially entails considerable code replication, since models using the same queryset definition need a private copy of said definition
- enforcing unique names for all model querysets requires retroactive renaming

## Rationale
The main purposes of th is decision are to 
- reduce the risk of modellers accidentally using querysets whose definitions have been modified without their knowledge
- make queryset definitions easy to find, locating them within the models they are being used for

## Feedback and Suggestions
Feedback is welcome.

---
