from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    LearnRSPN,
)

from matplotlib import pyplot as plt

from probabilistic_model.probabilistic_circuit.relational.rspns import (
    Person,
    Government,
    Supports,
    Nation,
    Region,
    Adjacent,
    Conflict,
    RSPNTemplate,
    class_spec_nation,
    class_spec_region,
)


def example():
    david = Person("David", 25)
    tom = Person("Tom", 27)
    checker_chan = Person("Simon Wallukat", 28)

    cdu = Government(funny=True)
    persons = [david, tom, checker_chan]
    supporters = [Supports(checker_chan, cdu)]
    n1 = Nation(government=cdu, supporters=supporters, persons=persons)

    daniel = Person("Daniel", 50)
    tede = Person("Tede", 25)
    daniel_union = Government(funny=False)

    knowrob_supporters = [daniel, tede]

    knowrob_nation = Nation(
        government=daniel_union,
        persons=[daniel, tede],
        supporters=[Supports(daniel, daniel_union), Supports(tede, daniel_union)],
        gdp=-10,
    )

    region = Region(
        nations=[n1, knowrob_nation],
        adjacency=[Adjacent(n1, knowrob_nation)],
        conflicts=[Conflict(n1, knowrob_nation)],
    )

    # n2 = let(Nation, [])
    # p3 = let(Person, [])
    # q = an(
    #     entity(
    #         n2,
    #         n2.government.funny == True,
    #         n2.persons == [david, tom, p3],
    #     )
    # )

    region_template = RSPNTemplate(class_spec=class_spec_region)
    region_template.probabilistic_circuit.plot_structure()
    plt.show()

    nation_template = RSPNTemplate(class_spec=class_spec_nation)
    nation_template.probabilistic_circuit.plot_structure()
    plt.show()

    grounded = region_template.ground(region)
    grounded.probabilistic_circuit.plot_structure()
    plt.show()

    grounded_nation = nation_template.ground(n1)
    grounded_nation.probabilistic_circuit.plot_structure()
    plt.show()

    # learned_nation = LearnRSPN(Region, region, class_spec_region)
    # learned_nation.probabilistic_circuit.plot_structure()
    # plt.show()

    learned_nation = LearnRSPN(Nation, [n1, knowrob_nation], class_spec_nation)
    learned_nation.probabilistic_circuit.plot_structure()
    plt.show()

    grounded_learned_nation = learned_nation.ground(n1)
    grounded_learned_nation.probabilistic_circuit.plot_structure()
    plt.show()


if __name__ == "__main__":
    example()
