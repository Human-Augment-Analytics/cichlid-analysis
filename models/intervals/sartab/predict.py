from models.common.predict import main, predict  # noqa: F401

NAMES = {'sartab': 'sartab'}

if __name__ == '__main__':
    main(NAMES, 'sartab', 'models.intervals.sartab.train')
